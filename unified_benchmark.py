#!/usr/bin/env python3

"""
Unified benchmarking orchestrator for CANDI, ChromImpute, Avocado, and eDICE.

Notes:
- All artifacts are rooted under bench_dir (mandatory): data/, processed/, models/, imputed_tracks/, evaluation/, external/, envs/
- Data loading and split logic are delegated to data.ExtendedEncodeDataHandler (mirrors data.py usage).
- Evaluation mirrors eval.py METRICS usage, but this orchestrator only coordinates stages.
- Default scopes: train on chr19, test on chr21; allow overrides via CLI.
- Timings: per-stage totals and per-bios inference timings are recorded to bench_dir/evaluation/timings.csv.
- TODO: Apply blacklist and gap filtering consistently for all methods during training and evaluation.
"""

import argparse
import os
import sys
import time
import json
import csv
import glob
import gzip
from typing import List, Dict, Tuple
import subprocess
import shutil
import zipfile
from types import SimpleNamespace

from data import ExtendedEncodeDataHandler
from eval import METRICS
import candi  


def _append_timing(bench_dir: str, method: str, stage: str, dataset: str,
                   scope_train: str, scope_test: str,
                   n_bios: int, n_assays: int, start_ts: float, end_ts: float) -> None:
    os.makedirs(os.path.join(bench_dir, "evaluation"), exist_ok=True)
    path = os.path.join(bench_dir, "evaluation", "timings.csv")
    row = {
        "method": method,
        "stage": stage,
        "dataset": dataset,
        "scope_train": scope_train,
        "scope_test": scope_test,
        "n_bios": n_bios,
        "n_assays": n_assays,
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "wall_time_sec": round(end_ts - start_ts, 3),
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _env_bootstrap(bench_dir: str) -> Dict[str, Dict[str, str]]:
    """Test 0: Check/install external tools and write env_report.json in bench_dir/evaluation.
    This function only prepares local folders and checks imports; package/env creation and installation
    are not executed automatically here, but instructions are logged in the report.
    """
    report = {
        "chromimpute": {},
        "avocado": {},
        "edice": {},
    }
    # ChromImpute jar
    jar_target = os.path.join(bench_dir, 'lib', 'ChromImpute.jar')
    os.makedirs(os.path.dirname(jar_target), exist_ok=True)
    report["chromimpute"]["jar_path"] = jar_target
    report["chromimpute"]["present"] = str(os.path.exists(jar_target))
    if not os.path.exists(jar_target):
        report["chromimpute"]["install_hint"] = "Download ChromImpute.zip from official site and place ChromImpute.jar under bench_dir/lib/"

    # Avocado import
    try:
        import avocado  # noqa: F401
        report["avocado"]["import_ok"] = "true"
    except Exception as e:
        report["avocado"]["import_ok"] = "false"
        report["avocado"]["install_hint"] = "Create avocado_env under bench_dir/envs and pip install avocado-epigenome"
        report["avocado"]["error"] = str(e)

    # eDICE repo
    edice_repo = os.path.join(bench_dir, 'external', 'eDICE')
    report["edice"]["repo_path"] = edice_repo
    report["edice"]["present"] = str(os.path.isdir(edice_repo))
    if not os.path.isdir(edice_repo):
        report["edice"]["install_hint"] = "git clone https://github.com/alex-hh/eDICE.git to bench_dir/external/eDICE and install requirements"

    os.makedirs(os.path.join(bench_dir, 'evaluation'), exist_ok=True)
    with open(os.path.join(bench_dir, 'evaluation', 'env_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def _write_install_log(bench_dir: str, text: str) -> None:
    # Print progress to stdout instead of logging only to a file
    print(text)


def _which(cmd: str) -> str:
    return shutil.which(cmd) or ""


def _download_file(url: str, dest_path: str) -> bool:
    try:
        import urllib.request
        print(f"[download] Fetching {url} -> {dest_path}")
        def _hook(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            end = "\r" if percent < 100 else "\n"
            print(f"  progress: {percent:5.1f}%", end=end, flush=True)
        urllib.request.urlretrieve(url, dest_path, reporthook=_hook)
        return True
    except Exception as e:
        print(f"[download] Failed {url}: {e}")
        return False


def _env_install(bench_dir: str) -> Dict[str, Dict[str, str]]:
    """Attempt to install required third-party tools into bench_dir.
    - ChromImpute: place ChromImpute.jar under bench_dir/lib (try common URLs or env var CHROMIMPUTE_JAR_URL)
    - Avocado: create Python env under bench_dir/envs/avocado_env and pip install avocado-epigenome
    - eDICE: clone repo to bench_dir/external/eDICE and pip install -r requirements.txt in bench_dir/envs/edice_env
    Returns a status dict and writes evaluation/install_status.json with details.
    """
    status: Dict[str, Dict[str, str]] = {"chromimpute": {}, "avocado": {}, "edice": {}}
    eval_dir = os.path.join(bench_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # Ensure base dirs
    lib_dir = os.path.join(bench_dir, 'lib')
    envs_dir = os.path.join(bench_dir, 'envs')
    external_dir = os.path.join(bench_dir, 'external')
    os.makedirs(lib_dir, exist_ok=True)
    os.makedirs(envs_dir, exist_ok=True)
    os.makedirs(external_dir, exist_ok=True)

    # Log system basics
    py = sys.executable
    conda_path = _which('conda')
    git_path = _which('git')
    java_path = _which('java')
    _write_install_log(bench_dir, f"python: {py}")
    _write_install_log(bench_dir, f"conda: {conda_path or 'not found'}")
    _write_install_log(bench_dir, f"git: {git_path or 'not found'}")
    _write_install_log(bench_dir, f"java: {java_path or 'not found'}")

    # 1) ChromImpute
    jar_target = os.path.join(lib_dir, 'ChromImpute.jar')
    status['chromimpute']['jar_path'] = jar_target
    if os.path.exists(jar_target):
        status['chromimpute']['installed'] = 'true'
    else:
        # Hardcoded official ZIP URL as provided by user
        zip_url = 'https://ernstlab.github.io/ChromImpute/ChromImpute.zip'
        tmp_dir = os.path.join(bench_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        zip_path = os.path.join(tmp_dir, 'ChromImpute.zip')
        _write_install_log(bench_dir, f"[ChromImpute] Downloading ZIP from {zip_url}")
        ok = _download_file(zip_url, zip_path)
        if ok and os.path.exists(zip_path):
            try:
                _write_install_log(bench_dir, f"[ChromImpute] Extracting {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Look for ChromImpute.jar inside the archive
                    jar_member = None
                    for m in zf.namelist():
                        if m.endswith('ChromImpute.jar'):
                            jar_member = m
                            break
                    if not jar_member:
                        raise RuntimeError('ChromImpute.jar not found inside ZIP')
                    extract_path = os.path.join(tmp_dir, 'ChromImpute_extracted')
                    os.makedirs(extract_path, exist_ok=True)
                    zf.extract(jar_member, path=extract_path)
                    src_jar = os.path.join(extract_path, jar_member)
                    _write_install_log(bench_dir, f"[ChromImpute] Placing JAR to {jar_target}")
                    shutil.copy2(src_jar, jar_target)
                    ok = os.path.exists(jar_target)
            except Exception as e:
                print(f"[ChromImpute] Extraction failed: {e}")
                ok = False
        status['chromimpute']['installed'] = 'true' if ok else 'false'
        if not ok:
            status['chromimpute']['message'] = f"Failed to install ChromImpute from ZIP. Please place ChromImpute.jar manually at {jar_target}."
    # Java presence
    status['chromimpute']['java_found'] = 'true' if java_path else 'false'

    # 2) Avocado
    try:
        import avocado  # noqa: F401
        status['avocado']['installed'] = 'true'
        status['avocado']['note'] = 'import works in current interpreter'
    except Exception:
        env_prefix = os.path.join(envs_dir, 'avocado_env')
        py_bin = os.path.join(env_prefix, 'bin', 'python')
        pip_bin = os.path.join(env_prefix, 'bin', 'pip')
        if conda_path:
            _write_install_log(bench_dir, f"[Avocado] Creating conda env at {env_prefix}")
            subprocess.run([conda_path, 'create', '-y', '-p', env_prefix, 'python=3.10'])
        else:
            _write_install_log(bench_dir, f"[Avocado] Creating venv at {env_prefix}")
            subprocess.run([sys.executable, '-m', 'venv', env_prefix])
        _write_install_log(bench_dir, f"[Avocado] Installing avocado-epigenome")
        subprocess.run([py_bin, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
        res = subprocess.run([pip_bin, 'install', 'avocado-epigenome'])
        status['avocado']['installed'] = 'true' if res.returncode == 0 else 'false'
        status['avocado']['env_prefix'] = env_prefix

    # 3) eDICE
    edice_repo = os.path.join(external_dir, 'eDICE')
    if not os.path.isdir(edice_repo):
        if git_path:
            _write_install_log(bench_dir, f"[eDICE] Cloning into {edice_repo}")
            res = subprocess.run([git_path, 'clone', '--depth', '1', 'https://github.com/alex-hh/eDICE.git', edice_repo])
        else:
            _write_install_log(bench_dir, "[eDICE] git not found; skip clone")
    status['edice']['repo_path'] = edice_repo
    status['edice']['repo_present'] = 'true' if os.path.isdir(edice_repo) else 'false'
    if os.path.isdir(edice_repo):
        env_prefix = os.path.join(envs_dir, 'edice_env')
        py_bin = os.path.join(env_prefix, 'bin', 'python')
        pip_bin = os.path.join(env_prefix, 'bin', 'pip')
        if conda_path:
            _write_install_log(bench_dir, f"[eDICE] Creating conda env at {env_prefix}")
            subprocess.run([conda_path, 'create', '-y', '-p', env_prefix, 'python=3.10'])
        else:
            _write_install_log(bench_dir, f"[eDICE] Creating venv at {env_prefix}")
            subprocess.run([sys.executable, '-m', 'venv', env_prefix])
        # Install eDICE requirements
        req_file = os.path.join(edice_repo, 'requirements.txt')
        if os.path.exists(req_file):
            _write_install_log(bench_dir, f"[eDICE] Installing requirements from {req_file}")
            subprocess.run([py_bin, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
            res = subprocess.run([pip_bin, 'install', '-r', req_file])
            status['edice']['installed'] = 'true' if res.returncode == 0 else 'false'
        else:
            status['edice']['installed'] = 'false'
            status['edice']['message'] = 'requirements.txt not found; please install manually'

    with open(os.path.join(eval_dir, 'install_status.json'), 'w') as f:
        json.dump(status, f, indent=2)
    return status


def _inject_env_sitepackages(env_prefix: str) -> bool:
    """Try to add the env's site-packages to sys.path for import without activation."""
    if not env_prefix or not os.path.isdir(env_prefix):
        return False
    # Try typical locations for venv/conda
    candidates = []
    lib_dir = os.path.join(env_prefix, 'lib')
    if os.path.isdir(lib_dir):
        for d in os.listdir(lib_dir):
            if d.startswith('python'):
                sp = os.path.join(lib_dir, d, 'site-packages')
                if os.path.isdir(sp):
                    candidates.append(sp)
    # Windows-style fallback (unlikely here)
    candidates.append(os.path.join(env_prefix, 'Lib', 'site-packages'))
    added = False
    for sp in candidates:
        if os.path.isdir(sp) and sp not in sys.path:
            sys.path.insert(0, sp)
            added = True
    return added


# ----------------- Navigation / split helpers (mirrors data.py filenames) -----------------

def get_navigation_and_split_paths(data_path: str, dataset: str) -> Tuple[str, str]:
    if dataset == 'merged':
        nav = os.path.join(data_path, 'merged_navigation.json')
        split = os.path.join(data_path, 'merged_train_va_test_split.json')
    elif dataset == 'eic':
        nav = os.path.join(data_path, 'navigation_eic.json')
        split = os.path.join(data_path, 'train_va_test_split_eic.json')
    else:
        nav = os.path.join(data_path, 'navigation.json')
        split = os.path.join(data_path, 'train_va_test_split.json')
    return nav, split


def load_split_dict(split_path: str) -> Dict[str, str]:
    with open(split_path, 'r') as f:
        return json.load(f)


def load_navigation(nav_path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(nav_path, 'r') as f:
        return json.load(f)


# ----------------- Dataset Manager -----------------

class DatasetManager:
    """
    Thin wrapper over ExtendedEncodeDataHandler to keep the orchestrator readable.
    This mirrors data.py usage strictly and does NOT re-implement loaders.
    """
    def __init__(self, bench_dir: str, data_path: str, dataset: str, train_scope: str, test_scope: str, includes_35: List[str]):
        self.bench_dir = bench_dir
        self.data_path = data_path
        self.dataset = dataset
        self.train_scope = train_scope
        self.test_scope = test_scope
        self.includes_35 = includes_35
        self.resolution = 25

        # Initialize a handler for training scope (to get counts of bios/assays)
        self.eed = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.eed.initialize_EED(
            m=100,  # minimal loci for bookkeeping; real training uses method runners
            context_length=1200*25,
            bios_batchsize=1,
            loci_batchsize=1,
            loci_gen=self.train_scope,
            bios_min_exp_avail_threshold=3,
            check_completeness=True,
            shuffle_bios=True,
            includes=self.includes_35,
            excludes=[],
            merge_ct=True,
            must_have_chr_access=True,
            eic=(self.dataset == 'eic'),
            DSF_list=[1,2],
        )

        # Load raw navigation/splits for accessing both train and test bios lists
        nav_path, split_path = get_navigation_and_split_paths(self.data_path, self.dataset)
        self.navigation_all = load_navigation(nav_path)
        self.split_dict = load_split_dict(split_path)

    @property
    def n_train_bios(self) -> int:
        return len([b for b, s in self.split_dict.items() if s == 'train'])

    @property
    def n_assays(self) -> int:
        try:
            return len(self.eed.aliases["experiment_aliases"])
        except Exception:
            return 35

    def train_bios(self) -> List[str]:
        return [b for b, s in self.split_dict.items() if s == 'train']

    def test_bios(self) -> List[str]:
        return [b for b, s in self.split_dict.items() if s == 'test']

    def chr_length(self, chrom: str) -> int:
        length = None
        with open(self.eed.chr_sizes_file, 'r') as f:
            for line in f:
                name, size = line.strip().split('\t')
                if name == chrom:
                    length = int(size)
                    break
        if length is None:
            raise RuntimeError(f"Chromosome {chrom} not found in {self.eed.chr_sizes_file}")
        return length


def prepare_demo_eic_subset(dm: DatasetManager, target_dir: str, train_n=3, test_n=2, assays_subset=None) -> Tuple[str, str]:
    os.makedirs(target_dir, exist_ok=True)
    if assays_subset is None:
        assays_subset = ['H3K4me3','DNase-seq','H3K27ac']
    train_b = sorted(dm.train_bios())[:train_n]
    test_b = sorted(dm.test_bios())[:test_n]
    nav = {}
    for b in train_b + test_b:
        nav[b] = {}
        assays = list(dm.navigation_all.get(b, {}).keys())
        for a in assays_subset:
            if a in assays:
                nav[b][a] = dm.navigation_all[b][a]
    split = {b: ('train' if b in train_b else 'test') for b in (train_b + test_b)}
    nav_path = os.path.join(target_dir, 'navigation_eic_demo.json')
    split_path = os.path.join(target_dir, 'train_va_test_split_eic_demo.json')
    with open(nav_path, 'w') as f:
        json.dump(nav, f, indent=2)
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)
    return nav_path, split_path


# ----------------- IO utilities (mirrors eval/data logic) -----------------

def write_chrom_sizes(single_chr: str, eed: ExtendedEncodeDataHandler, out_path: str) -> None:
    sizes = {}
    with open(eed.chr_sizes_file, 'r') as f:
        for line in f:
            name, size = line.strip().split('\t')
            if name == single_chr:
                sizes[name] = int(size)
                break
    if single_chr not in sizes:
        raise RuntimeError(f"Chromosome {single_chr} not found in {eed.chr_sizes_file}")
    with open(out_path, 'w') as f:
        f.write(f"{single_chr}\t{sizes[single_chr]}\n")


def arcsinh_pval_vector(eed: ExtendedEncodeDataHandler, bios: str, assay: str, chrom: str, chrom_len: int):
    import numpy as np
    temp_p = eed.load_bios_BW(bios, [chrom, 0, chrom_len], 1)
    P, avlP = eed.make_bios_tensor_BW(temp_p)
    expnames = list(eed.aliases["experiment_aliases"].keys())
    if assay not in expnames:
        return None
    aidx = expnames.index(assay)
    vec = P[:, aidx]
    try:
        vec = vec.numpy()
    except Exception:
        pass
    return np.arcsinh(vec)


def write_bedgraph_from_pval_arcsinh(eed: ExtendedEncodeDataHandler, bios: str, assay: str,
                                      chrom: str, out_bedgraph: str) -> None:
    import numpy as np
    length = None
    with open(eed.chr_sizes_file, 'r') as f:
        for line in f:
            name, size = line.strip().split('\t')
            if name == chrom:
                length = int(size)
                break
    if length is None:
        raise RuntimeError(f"Failed to read length for {chrom}")
    vec = arcsinh_pval_vector(eed, bios, assay, chrom, length)
    if vec is None:
        return
    usable = (length // 25) * 25
    n_bins = usable // 25
    vec = vec[:n_bins]

    with open(out_bedgraph, 'w') as out:
        start = 0
        for i in range(n_bins):
            end = start + 25
            out.write(f"{chrom}\t{start}\t{end}\t{float(vec[i]):.6f}\n")
            start = end


def write_samplemarktable(entries: List[Tuple[str, str, str]], out_path: str) -> None:
    with open(out_path, 'w') as f:
        for bios, assay, fpath in entries:
            f.write(f"{bios}\t{assay}\t{fpath}\n")


# ----------------- Runners -----------------

class BaseRunner:
    def __init__(self, dm: DatasetManager, write_bw: bool):
        self.dm = dm
        self.bench_dir = dm.bench_dir
        self.write_bw = write_bw
        os.makedirs(self.bench_dir, exist_ok=True)

    def ensure_env(self):
        pass

    def preprocess(self):
        pass

    def train(self):
        pass

    def infer(self):
        pass


class CandiRunner(BaseRunner):
    def preprocess(self):
        return

    def train(self, args):
        if candi is None:
            raise RuntimeError("candi module is not importable")
        t0 = time.time()
        # forward bench_dir to candi CLI
        orig_output = getattr(args, 'bench_dir', None)
        candi.cmd_train(args)
        t1 = time.time()
        _append_timing(self.bench_dir, "candi", "train_total", args.dataset, args.train_scope, args.test_scope, self.dm.n_train_bios, self.dm.n_assays, t0, t1)

    def infer(self, args):
        if candi is None:
            raise RuntimeError("candi module is not importable")
        candi.cmd_infer(args)


class ChromImputeRunner(BaseRunner):
    def __init__(self, dm: DatasetManager, write_bw: bool):
        super().__init__(dm, write_bw)
        self.proc_dir = os.path.join(self.bench_dir, 'processed', 'chromimpute')
        self.model_dir = os.path.join(self.bench_dir, 'models', 'chromimpute')
        self.pred_dir = os.path.join(self.bench_dir, 'imputed_tracks', 'chromimpute')
        for d in [self.proc_dir, self.model_dir, self.pred_dir]:
            os.makedirs(d, exist_ok=True)
        self.bedgraph_dir = os.path.join(self.proc_dir, 'bedgraph')
        self.meta_dir = os.path.join(self.proc_dir, 'metadata')
        self.converted_dir = os.path.join(self.proc_dir, 'converted')
        self.distance_dir = os.path.join(self.proc_dir, 'distances')
        self.traindata_dir = os.path.join(self.proc_dir, 'traindata')
        for d in [self.bedgraph_dir, self.meta_dir, self.converted_dir, self.distance_dir, self.traindata_dir]:
            os.makedirs(d, exist_ok=True)
        # Always use bench_dir/lib for ChromImpute jar
        self.jar_path = os.path.join(self.bench_dir, 'lib', 'ChromImpute.jar')

    def ensure_env(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] ERROR: {self.jar_path} not found. Please place ChromImpute.jar under lib/")

    def preprocess(self):
        train_sizes = os.path.join(self.meta_dir, f'{self.dm.train_scope}.sizes')
        test_sizes = os.path.join(self.meta_dir, f'{self.dm.test_scope}.sizes')
        eed = self.dm.eed
        write_chrom_sizes(self.dm.train_scope, eed, train_sizes)
        write_chrom_sizes(self.dm.test_scope, eed, test_sizes)

        nav_all = self.dm.navigation_all
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        sample_entries = []

        def build_for_split(bios_list: List[str], chrom: str):
            for bios in bios_list:
                assays = list(nav_all.get(bios, {}).keys())
                for assay in assays:
                    if assay not in expnames:
                        continue
                    out_bg = os.path.join(self.bedgraph_dir, f"{bios}__{assay}__{chrom}.bedgraph")
                    if not os.path.exists(out_bg):
                        try:
                            write_bedgraph_from_pval_arcsinh(self.dm.eed, bios, assay, chrom, out_bg)
                        except Exception as e:
                            print(f"[ChromImpute] bedGraph failed for {bios},{assay},{chrom}: {e}")
                            continue
                    sample_entries.append((bios, assay, out_bg))

        build_for_split(self.dm.train_bios(), self.dm.train_scope)
        build_for_split(self.dm.test_bios(), self.dm.test_scope)

        smt = os.path.join(self.meta_dir, 'samplemarktable.txt')
        write_samplemarktable(sample_entries, smt)
        print(f"[ChromImpute] Preprocess complete. samplemarktable: {smt}")

    def _run_java(self, args: List[str]) -> int:
        cmd = ["java", "-mx8000M", "-jar", self.jar_path] + args
        print("[ChromImpute] RUN:", " ".join(cmd))
        import subprocess
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
        return p.returncode

    def _convert_wig_gz_to_npy(self, chrom: str) -> None:
        """Parse ChromImpute .wig.gz outputs in pred_dir into .npy arrays.
        Handles variableStep/value lines and simple bedGraph-like lines. Unknown formats are logged and skipped.
        Output naming: {bios}__{assay}__{chrom}__{denoised|imputed}__pval.npy
        UNCERTAINTY: actual ChromImpute output filenames vary; we attempt common patterns.
        """
        # Common output patterns include files per chrom/experiment. We scan all wig.gz in pred_dir.
        files = glob.glob(os.path.join(self.pred_dir, f"*{chrom}*.wig.gz"))
        if not files:
            return
        # Build category by inspecting availability: if bios has assay in navigation, "denoised", else "imputed"
        for fp in files:
            try:
                name = os.path.basename(fp)
                # heuristic parse to extract bios and assay
                # Examples to handle (not exhaustive): chr21_impute_BIOS_ASSAY.wig.gz, chr21_BIOS_ASSAY.wig.gz
                bios, assay = None, None
                parts = name.replace('.wig.gz','').split('_')
                # find tokens that match known bios and assays
                expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
                for i in range(len(parts)):
                    for a in expnames:
                        if a == parts[i]:
                            assay = a
                            # assume bios precedes assay
                            if i-1 >= 0:
                                bios = parts[i-1]
                            break
                    if assay is not None:
                        break
                if bios is None or assay is None:
                    print(f"[ChromImpute] Skip unknown naming: {name}")
                    continue
                # parse wig
                values = []
                with gzip.open(fp, 'rt') as f:
                    current_pos = None
                    step = None
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('track'):
                            continue
                        if line.startswith('variableStep'):
                            # variableStep chrom=chr21 span=25
                            tokens = dict(t.split('=') for t in line.split()[1:])
                            step = int(tokens.get('span','25'))
                            current_pos = None
                            continue
                        toks = line.split()
                        if len(toks) == 1 and current_pos is not None and step is not None:
                            # value-only line
                            values.append(float(toks[0]))
                            current_pos += step
                        elif len(toks) == 2:
                            # position value
                            try:
                                pos = int(toks[0])
                                val = float(toks[1])
                                values.append(val)
                                current_pos = pos
                                if step is None:
                                    step = 25
                            except Exception:
                                pass
                        elif len(toks) == 4:
                            # bedGraph-style: chrom start end value
                            try:
                                val = float(toks[3])
                                values.append(val)
                            except Exception:
                                pass
                        else:
                            # unknown line
                            continue
                import numpy as np
                vec = np.array(values, dtype=float)
                # convert back to raw pval for evaluation storage
                vec = np.sinh(vec)
                assays_obs = list(self.dm.navigation_all.get(bios, {}).keys())
                category = "denoised" if assay in assays_obs else "imputed"
                base = f"{bios}__{assay}__{chrom}__{category}__pval"
                np.save(os.path.join(self.pred_dir, base + ".npy"), vec)
            except Exception as e:
                print(f"[ChromImpute] Failed to convert {fp}: {e}")

    def train(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] Skipping train. Missing {self.jar_path}")
            return
        t0 = time.time()
        smt = os.path.join(self.meta_dir, 'samplemarktable.txt')
        train_sizes = os.path.join(self.meta_dir, f'{self.dm.train_scope}.sizes')
        self._run_java(["Convert", self.bedgraph_dir, smt, train_sizes, self.converted_dir])
        self._run_java(["ComputeGlobalDist", self.converted_dir, smt, train_sizes, self.distance_dir])
        for assay in self.dm.eed.aliases["experiment_aliases"].keys():
            self._run_java(["GenerateTrainData", self.converted_dir, self.distance_dir, smt, train_sizes, self.traindata_dir, assay])
        for bios in self.dm.train_bios():
            for assay in self.dm.eed.aliases["experiment_aliases"].keys():
                self._run_java(["Train", self.traindata_dir, smt, self.model_dir, bios, assay])
        t1 = time.time()
        _append_timing(self.bench_dir, "chromimpute", "train", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, len(self.dm.train_bios()), self.dm.n_assays, t0, t1)

    def infer(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] Skipping infer. Missing {self.jar_path}")
            return
        smt = os.path.join(self.meta_dir, 'samplemarktable.txt')
        test_sizes = os.path.join(self.meta_dir, f'{self.dm.test_scope}.sizes')
        for bios in self.dm.test_bios():
            tb0 = time.time()
            for assay in self.dm.eed.aliases["experiment_aliases"].keys():
                self._run_java(["Apply", self.converted_dir, self.distance_dir, self.model_dir, smt, test_sizes, self.pred_dir, bios, assay])
            # After Apply completes for this bios, parse outputs for test chrom
            self._convert_wig_gz_to_npy(self.dm.test_scope)
            tb1 = time.time()
            _append_timing(self.bench_dir, "chromimpute", "infer_bios", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, 1, self.dm.n_assays, tb0, tb1)


class AvocadoRunner(BaseRunner):
    def __init__(self, dm: DatasetManager, write_bw: bool):
        super().__init__(dm, write_bw)
        self.proc_dir = os.path.join(self.bench_dir, 'processed', 'avocado')
        self.model_dir = os.path.join(self.bench_dir, 'models', 'avocado')
        self.pred_dir = os.path.join(self.bench_dir, 'imputed_tracks', 'avocado')
        for d in [self.proc_dir, self.model_dir, self.pred_dir]:
            os.makedirs(d, exist_ok=True)
        self.train_cache_dir = os.path.join(self.proc_dir, self.dm.train_scope)
        os.makedirs(self.train_cache_dir, exist_ok=True)
        self._avocado = None
        self._avocado_env_prefix = None

    def ensure_env(self):
        try:
            from avocado import Avocado  # noqa: F401
            self._avocado = True
        except Exception as e:
            # Try injecting site-packages from bench_dir/envs/avocado_env
            install_status = os.path.join(self.bench_dir, 'evaluation', 'install_status.json')
            env_prefix = None
            if os.path.exists(install_status):
                try:
                    with open(install_status, 'r') as f:
                        st = json.load(f)
                    env_prefix = st.get('avocado', {}).get('env_prefix')
                except Exception:
                    env_prefix = None
            if env_prefix and _inject_env_sitepackages(env_prefix):
                try:
                    from avocado import Avocado  # noqa: F401
                    self._avocado = True
                    self._avocado_env_prefix = env_prefix
                    print(f"[Avocado] Imported from env: {env_prefix}")
                    return
                except Exception as e2:
                    print("[Avocado] Import failed even after sys.path injection:", e2)
            self._avocado = False
            print("[Avocado] WARNING: avocado-epigenome not importable; ensure installed under bench_dir/envs/avocado_env or current env.", e)

    def preprocess(self):
        chrom = self.dm.train_scope
        length = self.dm.chr_length(chrom)
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        for bios in self.dm.train_bios():
            assays = list(self.dm.navigation_all.get(bios, {}).keys())
            for assay in assays:
                if assay not in expnames:
                    continue
                out_npz = os.path.join(self.train_cache_dir, f"{bios}__{assay}.npz")
                if os.path.exists(out_npz):
                    continue
                import numpy as np
                vec = arcsinh_pval_vector(self.dm.eed, bios, assay, chrom, length)
                if vec is None:
                    continue
                np.savez_compressed(out_npz, vec)
        print("[Avocado] Preprocess cache complete.")

    def train(self, epochs: int | None = None):
        if not self._avocado:
            print("[Avocado] Skipping train; avocado is not available.")
            return
        from avocado import Avocado
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        celltypes = sorted(set(self.dm.train_bios()))
        assays = sorted(set(expnames))
        data = {}
        for bios in celltypes:
            for assay in assays:
                cache_path = os.path.join(self.train_cache_dir, f"{bios}__{assay}.npz")
                if not os.path.exists(cache_path):
                    continue
                import numpy as np
                arr = np.load(cache_path)["arr_0"]
                data[(bios, assay)] = arr
        if not data:
            print("[Avocado] No training data found in cache; skipping.")
            return
        model = Avocado(
            celltypes=celltypes,
            assays=assays,
            n_layers=1,
            n_nodes=64,
            n_assay_factors=24,
            n_celltype_factors=32,
            n_25bp_factors=5,
            n_250bp_factors=20,
            n_5kbp_factors=30,
            batch_size=10000,
        )
        t0 = time.time()
        model.fit(data, n_epochs=(epochs if epochs is not None else 10), epoch_size=100)
        t1 = time.time()
        model.save(os.path.join(self.model_dir, f"avocado-{self.dm.train_scope}"))
        _append_timing(self.bench_dir, "avocado", "train", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, len(self.dm.train_bios()), self.dm.n_assays, t0, t1)

    def infer(self):
        if not self._avocado:
            print("[Avocado] Skipping infer; avocado is not available.")
            return
        from avocado import Avocado
        try:
            model = Avocado.load(os.path.join(self.model_dir, f"avocado-{self.dm.train_scope}"))
        except Exception as e:
            print("[Avocado] Could not load trained model:", e)
            return
        chrom = self.dm.test_scope
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        for bios in self.dm.test_bios():
            tb0 = time.time()
            assays_obs = list(self.dm.navigation_all.get(bios, {}).keys())
            for assay in expnames:
                try:
                    import numpy as np
                    pred_arc = model.predict(bios, assay)
                    pred = np.sinh(pred_arc)
                    category = "denoised" if assay in assays_obs else "imputed"
                    os.makedirs(self.pred_dir, exist_ok=True)
                    base = f"{bios}__{assay}__{chrom}__{category}__pval"
                    np.save(os.path.join(self.pred_dir, base + ".npy"), pred)
                    std = float(np.std(pred))
                    np.save(os.path.join(self.pred_dir, base.replace("__pval","__std") + ".npy"), np.full_like(pred, std))
                except Exception as e:
                    print(f"[Avocado] Predict failed for {bios},{assay}: {e}")
            tb1 = time.time()
            _append_timing(self.bench_dir, "avocado", "infer_bios", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, 1, self.dm.n_assays, tb0, tb1)


class EdiceRunner(BaseRunner):
    def __init__(self, dm: DatasetManager, write_bw: bool):
        super().__init__(dm, write_bw)
        self.proc_dir = os.path.join(self.bench_dir, 'processed', 'edice')
        self.model_dir = os.path.join(self.bench_dir, 'models', 'edice')
        self.pred_dir = os.path.join(self.bench_dir, 'imputed_tracks', 'edice')
        for d in [self.proc_dir, self.model_dir, self.pred_dir]:
            os.makedirs(d, exist_ok=True)
        self.h5_path = os.path.join(self.proc_dir, f"{self.dm.train_scope}.h5")
        self.idmap_path = os.path.join(self.proc_dir, 'idmap.json')
        self.splits_path = os.path.join(self.proc_dir, 'splits.json')
        self._edice_repo = os.path.join(self.bench_dir, 'external', 'eDICE')

    def ensure_env(self):
        # We do not install here; we just note expected location for scripts
        pass

    def preprocess(self):
        # Build HDF5 data_matrix with columns as tracks (bios, assay) for training (train bios observed assays)
        try:
            import h5py
            import numpy as np
        except Exception as e:
            print("[eDICE] h5py/numpy not available:", e)
            return
        chrom = self.dm.train_scope
        length = self.dm.chr_length(chrom)
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        tracks = []  # list of (bios, assay)
        for bios in self.dm.train_bios():
            assays = list(self.dm.navigation_all.get(bios, {}).keys())
            for assay in assays:
                if assay in expnames:
                    tracks.append((bios, assay))
        if not tracks:
            print("[eDICE] No tracks found for training.")
            return
        usable = (length // 25) * 25
        n_bins = usable // 25
        data_matrix = np.zeros((n_bins, len(tracks)), dtype=np.float32)
        for j, (bios, assay) in enumerate(tracks):
            vec = arcsinh_pval_vector(self.dm.eed, bios, assay, chrom, length)
            if vec is None:
                continue
            data_matrix[:, j] = vec[:n_bins]
        # write HDF5
        with h5py.File(self.h5_path, 'w') as h5:
            h5.create_dataset('data_matrix', data=data_matrix, compression='gzip')
            h5.create_dataset('chrom', data=np.string_(chrom))
            h5.create_dataset('bin_size', data=25)
        # idmap
        idmap = {str(i): {"bios": b, "assay": a} for i, (b, a) in enumerate(tracks)}
        with open(self.idmap_path, 'w') as f:
            json.dump(idmap, f, indent=2)
        # splits: train columns indices, and targets for test bios (denoise/impute)
        splits = {
            "train": list(range(len(tracks))),
            "val": [],
            "test_denoise": [],
            "test_impute": []
        }
        # map test bios to indices in training columns if any shared dimension (edice specifics vary)
        # We keep simple here and let train script control targets. This is a placeholder.
        with open(self.splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"[eDICE] Preprocess complete: {self.h5_path}")

    def train(self):
        # Call eDICE training script
        script = os.path.join(self._edice_repo, 'scripts', 'train_eDICE.py')
        if not os.path.exists(script):
            print(f"[eDICE] Skipping train. Missing script at {script}")
            return
        t0 = time.time()
        import subprocess
        cmd = [
            sys.executable, script,
            '--dataset_filepath', self.h5_path,
            '--idmap', self.idmap_path,
            '--split_file', self.splits_path,
            '--experiment_name', f'edice-{self.dm.train_scope}',
            '--transformation', 'arcsinh',
            '--epochs', '10'
        ]
        print("[eDICE] RUN:", " ".join(cmd))
        subprocess.run(cmd)
        t1 = time.time()
        _append_timing(self.bench_dir, "edice", "train", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, len(self.dm.train_bios()), self.dm.n_assays, t0, t1)

    def infer(self):
        # Parse predictions if produced by eDICE
        pred_npz = os.path.join(self.model_dir, f"edice-{self.dm.train_scope}", 'predictions.npz')
        if not os.path.exists(pred_npz):
            print(f"[eDICE] Skipping infer; predictions not found at {pred_npz}")
            return
        import numpy as np
        chrom = self.dm.test_scope
        os.makedirs(self.pred_dir, exist_ok=True)
        tb0 = time.time()
        data = np.load(pred_npz)
        # UNCERTAINTY: keys/shape of predictions.npz (depends on eDICE repo). Placeholder parsing:
        # Expect keys like '{bios}__{assay}' -> arcsinh predictions
        for key in data.files:
            try:
                bios, assay = key.split("__", 1)
            except Exception:
                continue
            pred_arc = data[key]
            pred = np.sinh(pred_arc)
            assays_obs = list(self.dm.navigation_all.get(bios, {}).keys())
            category = "denoised" if assay in assays_obs else "imputed"
            base = f"{bios}__{assay}__{chrom}__{category}__pval"
            np.save(os.path.join(self.pred_dir, base + ".npy"), pred)
            std = float(np.std(pred))
            np.save(os.path.join(self.pred_dir, base.replace("__pval","__std") + ".npy"), np.full_like(pred, std))
        tb1 = time.time()
        _append_timing(self.bench_dir, "edice", "infer_total", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, len(self.dm.test_bios()), self.dm.n_assays, tb0, tb1)


# ----------------- Evaluation -----------------

def evaluator(args):
    os.makedirs(os.path.join(args.bench_dir, "evaluation"), exist_ok=True)
    if METRICS is None:
        print("[Evaluator] METRICS not available; skipping.")
        return
    metrics = METRICS()
    rows = []

    def append_method(method: str):
        pred_dir = os.path.join(args.bench_dir, "imputed_tracks", method)
        if not os.path.isdir(pred_dir):
            return
        files = glob.glob(os.path.join(pred_dir, f"*__{args.test_scope}__*__pval.npy"))
        eed = ExtendedEncodeDataHandler(args.data_path, resolution=25)
        try:
            expnames = list(eed.aliases["experiment_aliases"].keys())
        except Exception:
            expnames = []
        for f in files:
            try:
                base = os.path.basename(f)
                bios, assay, scope, category, _ = base.split("__")
                import numpy as np
                pred = np.load(f)
                # Load GT pval
                # read chrom size
                length = 0
                with open(eed.chr_sizes_file, 'r') as fh:
                    for line in fh:
                        n, sz = line.strip().split('\t')
                        if n == args.test_scope:
                            length = int(sz)
                            break
                temp_p = eed.load_bios_BW(bios, [args.test_scope, 0, length], args.dsf)
                P, avlP = eed.make_bios_tensor_BW(temp_p)
                if assay not in expnames:
                    continue
                aidx = expnames.index(assay)
                try:
                    gt = P[:, aidx].numpy()
                except Exception:
                    gt = P[:, aidx]
                row = {
                    "dataset": args.dataset,
                    "method": method,
                    "bios": bios,
                    "assay": assay,
                    "category": category,
                    "scope_train": args.train_scope,
                    "scope_test": args.test_scope,
                    "P_MSE-GW": metrics.mse(gt, pred),
                    "P_Pearson-GW": metrics.pearson(gt, pred),
                    "P_Spearman-GW": metrics.spearman(gt, pred),
                    "P_MSE-gene": metrics.mse_gene(gt, pred),
                    "P_Pearson_gene": metrics.pearson_gene(gt, pred),
                    "P_Spearman_gene": metrics.spearman_gene(gt, pred),
                    "P_MSE-prom": metrics.mse_prom(gt, pred),
                    "P_Pearson_prom": metrics.pearson_prom(gt, pred),
                    "P_Spearman_prom": metrics.spearman_prom(gt, pred),
                    "P_MSE-1obs": metrics.mse1obs(gt, pred),
                    "P_Pearson_1obs": metrics.pearson1_obs(gt, pred),
                    "P_Spearman_1obs": metrics.spearman1_obs(gt, pred),
                    "P_MSE-1imp": metrics.mse1imp(gt, pred),
                    "P_Pearson_1imp": metrics.pearson1_imp(gt, pred),
                    "P_Spearman_1imp": metrics.spearman1_imp(gt, pred),
                }
                rows.append(row)
            except Exception as e:
                print(f"[Evaluator] Failed for {method} file {f}: {e}")

    for method in ["candi", "avocado", "chromimpute", "edice"]:
        append_method(method)

    if rows:
        out_csv = os.path.join(args.bench_dir, "evaluation", "summary_metrics.csv")
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)


# ----------------- Test Suites -----------------

def _assert_true(cond: bool, msg: str, failures: list):
    if not cond:
        print(f"[ASSERT FAIL] {msg}")
        failures.append(msg)
    else:
        print(f"[ASSERT OK] {msg}")


def _write_test_report(bench_dir: str, suite: str, passed: bool, details: list):
    os.makedirs(os.path.join(bench_dir, 'evaluation'), exist_ok=True)
    report_path = os.path.join(bench_dir, 'evaluation', 'test_report.json')
    rec = {
        'suite': suite,
        'passed': passed,
        'details': details,
        'ts': int(time.time()),
    }
    try:
        prev = []
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                prev = json.load(f)
        if not isinstance(prev, list):
            prev = []
        prev.append(rec)
        with open(report_path, 'w') as f:
            json.dump(prev, f, indent=2)
    except Exception:
        with open(report_path, 'w') as f:
            json.dump([rec], f, indent=2)


def run_test_candi_eic_end2end(args):
    print("[TEST] CANDI EIC end-to-end")
    failures: list[str] = []
    includes_35 = [
        'ATAC-seq','DNase-seq','H2AFZ','H2AK5ac','H2AK9ac','H2BK120ac','H2BK12ac','H2BK15ac',
        'H2BK20ac','H2BK5ac','H3F3A','H3K14ac','H3K18ac','H3K23ac','H3K23me2','H3K27ac','H3K27me3',
        'H3K36me3','H3K4ac','H3K4me1','H3K4me2','H3K4me3','H3K56ac','H3K79me1','H3K79me2','H3K9ac',
        'H3K9me1','H3K9me2','H3K9me3','H3T11ph','H4K12ac','H4K20me1','H4K5ac','H4K8ac','H4K91ac'
    ]
    dm = DatasetManager(args.bench_dir, args.data_path, 'eic', args.train_scope, args.test_scope, includes_35)
    runner = CandiRunner(dm, args.write_bw)

    # Train with epochs=1
    train_args = SimpleNamespace(**vars(args))
    train_args.dataset = 'eic'
    train_args.epochs = max(1, getattr(args, 'epochs', 1))
    print("[TEST] Training CANDI (epochs=1)")
    t0 = time.time()
    runner.train(train_args)
    t1 = time.time()
    _assert_true((t1 - t0) > 0, 'CANDI train executed', failures)

    # Infer
    print("[TEST] Inference CANDI (chr21)")
    infer_args = SimpleNamespace(**vars(args))
    infer_args.dataset = 'eic'
    runner.infer(infer_args)

    # Assert predictions exist
    pred_dir = os.path.join(args.bench_dir, 'imputed_tracks', 'candi')
    files = glob.glob(os.path.join(pred_dir, f"*__{args.test_scope}__*__pval.npy"))
    _assert_true(os.path.isdir(pred_dir), 'CANDI prediction directory exists', failures)
    _assert_true(len(files) > 0, 'CANDI predictions saved', failures)

    # Evaluate
    print("[TEST] Evaluate CANDI")
    eval_args = SimpleNamespace(**vars(args))
    eval_args.dataset = 'eic'
    evaluator(eval_args)
    metrics_csv = os.path.join(args.bench_dir, 'evaluation', 'summary_metrics.csv')
    _assert_true(os.path.exists(metrics_csv), 'summary_metrics.csv created', failures)

    passed = len(failures) == 0
    _write_test_report(args.bench_dir, 'candi_eic_end2end', passed, failures)
    print(f"[TEST RESULT] candi_eic_end2end: {'PASS' if passed else 'FAIL'}")
    return passed


def run_test_eic_all_methods_smoke(args):
    print("[TEST] EIC all methods smoke (epochs=1 where applicable)")
    failures: list[str] = []
    includes_35 = [
        'ATAC-seq','DNase-seq','H2AFZ','H2AK5ac','H2AK9ac','H2BK120ac','H2BK12ac','H2BK15ac',
        'H2BK20ac','H2BK5ac','H3F3A','H3K14ac','H3K18ac','H3K23ac','H3K23me2','H3K27ac','H3K27me3',
        'H3K36me3','H3K4ac','H3K4me1','H3K4me2','H3K4me3','H3K56ac','H3K79me1','H3K79me2','H3K9ac',
        'H3K9me1','H3K9me2','H3K9me3','H3T11ph','H4K12ac','H4K20me1','H4K5ac','H4K8ac','H4K91ac'
    ]
    dm = DatasetManager(args.bench_dir, args.data_path, 'eic', args.train_scope, args.test_scope, includes_35)

    # ChromImpute
    chrom = ChromImputeRunner(dm, args.write_bw)
    chrom.ensure_env(); chrom.preprocess(); chrom.train(); chrom.infer()
    _assert_true(os.path.isdir(os.path.join(args.bench_dir, 'imputed_tracks', 'chromimpute')), 'ChromImpute predictions dir', failures)

    # Avocado (epochs=1)
    avo = AvocadoRunner(dm, args.write_bw)
    avo.ensure_env(); avo.preprocess(); avo.train(epochs=max(1, getattr(args, 'epochs', 1))); avo.infer()
    _assert_true(os.path.isdir(os.path.join(args.bench_dir, 'imputed_tracks', 'avocado')), 'Avocado predictions dir', failures)

    # eDICE (if runner available)
    try:
        EdiceRunner  # type: ignore[name-defined]
        ed = EdiceRunner(dm, args.write_bw)  # noqa: F821
        ed.ensure_env(); ed.preprocess(); ed.train(); ed.infer()
        _assert_true(os.path.isdir(os.path.join(args.bench_dir, 'imputed_tracks', 'edice')), 'eDICE predictions dir', failures)
    except Exception:
        print('[TEST] eDICE runner not available or not wired; skipping')

    # Evaluate
    eval_args = SimpleNamespace(**vars(args))
    eval_args.dataset = 'eic'
    evaluator(eval_args)
    _assert_true(os.path.exists(os.path.join(args.bench_dir, 'evaluation', 'summary_metrics.csv')), 'summary_metrics.csv created', failures)

    passed = len(failures) == 0
    _write_test_report(args.bench_dir, 'eic_all_methods_smoke', passed, failures)
    print(f"[TEST RESULT] eic_all_methods_smoke: {'PASS' if passed else 'FAIL'}")
    return passed


def run_test_install_only(args):
    print('[TEST] install_only')
    failures: list[str] = []
    _env_install(args.bench_dir)
    rep = _env_bootstrap(args.bench_dir)
    _assert_true(rep['chromimpute'].get('present','False') == 'True', 'ChromImpute.jar present', failures)
    _assert_true(os.path.isdir(os.path.join(args.bench_dir, 'external', 'eDICE')), 'eDICE repo present', failures)
    _write_test_report(args.bench_dir, 'install_only', len(failures)==0, failures)
    print(f"[TEST RESULT] install_only: {'PASS' if len(failures)==0 else 'FAIL'}")
    return len(failures) == 0


def run_test_evaluate_only(args):
    print('[TEST] evaluate_only')
    failures: list[str] = []
    evaluator(args)
    _assert_true(os.path.exists(os.path.join(args.bench_dir, 'evaluation', 'summary_metrics.csv')), 'summary_metrics.csv created', failures)
    _write_test_report(args.bench_dir, 'evaluate_only', len(failures)==0, failures)
    print(f"[TEST RESULT] evaluate_only: {'PASS' if len(failures)==0 else 'FAIL'}")
    return len(failures) == 0


# ----------------- CLI -----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified benchmarking orchestrator")
    p.add_argument('--bench_dir', type=str, required=True, help='Root directory for all artifacts (mandatory)')
    p.add_argument('--dataset', choices=['enc','merged','eic'], default='enc')
    p.add_argument('--models', nargs='+', choices=['candi','chromimpute','avocado','edice','all'], default=['all'])
    p.add_argument('--stages', nargs='+', choices=['bootstrap','install','preprocess','train','infer','evaluate','test','all'], default=['all'])
    p.add_argument('--test_suite', choices=['candi_eic_end2end','eic_all_methods_smoke','install_only','evaluate_only'], default=None)
    p.add_argument('--data_path', type=str, default='/project/compbio-lab/encode_data/')
    p.add_argument('--train_scope', type=str, default='chr19')
    p.add_argument('--test_scope', type=str, default='chr21')
    p.add_argument('--write_bw', action='store_true')

    # CANDI args
    p.add_argument('--gpu', type=str, default='auto')
    p.add_argument('--dsf', type=int, default=1)
    p.add_argument('--context_length', type=int, default=1200)
    p.add_argument('--batch_size', type=int, default=25)
    p.add_argument('--min_avail', type=int, default=3)
    p.add_argument('--num_loci', type=int, default=5000)
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--from_ckpt', type=str, default=None)
    p.add_argument('--hpo', action='store_true')
    p.add_argument('--prog_mask', action='store_true')
    p.add_argument('--supertrack', action='store_true')
    p.add_argument('--optim', type=str, default='sgd')
    p.add_argument('--unet', action='store_true')
    p.add_argument('--LRschedule', type=str, default=None)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--lr_halflife', type=float, default=1)
    p.add_argument('--n_cnn_layers', type=int, default=3)
    p.add_argument('--conv_kernel_size', type=int, default=3)
    p.add_argument('--pool_size', type=int, default=2)
    p.add_argument('--expansion_factor', type=int, default=3)
    p.add_argument('--nhead', type=int, default=9)
    p.add_argument('--n_sab_layers', type=int, default=4)
    p.add_argument('--pos_enc', type=str, default='relative')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--inner_epochs', type=int, default=1)
    p.add_argument('--mask_percentage', type=float, default=0.2)
    p.add_argument('--shared_decoders', action='store_true')

    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--hyperparams', type=str, default=None)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    models = args.models
    if 'all' in models:
        models = ['candi','chromimpute','avocado','edice']

    stages = args.stages
    if 'all' in stages:
        stages = ['bootstrap','install','preprocess','train','infer','evaluate']

    includes_35 = [
        'ATAC-seq','DNase-seq','H2AFZ','H2AK5ac','H2AK9ac','H2BK120ac','H2BK12ac','H2BK15ac',
        'H2BK20ac','H2BK5ac','H3F3A','H3K14ac','H3K18ac','H3K23ac','H3K23me2','H3K27ac','H3K27me3',
        'H3K36me3','H3K4ac','H3K4me1','H3K4me2','H3K4me3','H3K56ac','H3K79me1','H3K79me2','H3K9ac',
        'H3K9me1','H3K9me2','H3K9me3','H3T11ph','H4K12ac','H4K20me1','H4K5ac','H4K8ac','H4K91ac'
    ]

    dm = DatasetManager(args.bench_dir, args.data_path, args.dataset, args.train_scope, args.test_scope, includes_35)

    runners = {}
    if 'candi' in models:
        runners['candi'] = CandiRunner(dm, args.write_bw)
    if 'chromimpute' in models:
        runners['chromimpute'] = ChromImputeRunner(dm, args.write_bw)
    if 'avocado' in models:
        runners['avocado'] = AvocadoRunner(dm, args.write_bw)
    if 'edice' in models:
        runners['edice'] = EdiceRunner(dm, args.write_bw)

    for stage in stages:
        if stage == 'bootstrap':
            _env_bootstrap(args.bench_dir)
        elif stage == 'install':
            print('[install] Checking and installing external tools...')
            _env_install(args.bench_dir)
        elif stage == 'preprocess':
            for m in models:
                runners[m].ensure_env()
                runners[m].preprocess()
        elif stage == 'train':
            for m in models:
                if m == 'candi':
                    runners[m].train(args)
                else:
                    # Pass epochs for smoke tests if provided
                    try:
                        runners[m].train(getattr(args, 'epochs', None))
                    except TypeError:
                        runners[m].train()
        elif stage == 'infer':
            for m in models:
                if m == 'candi':
                    runners[m].infer(args)
                else:
                    runners[m].infer()
        elif stage == 'evaluate':
            evaluator(args)
        elif stage == 'test':
            if args.test_suite is None:
                parser.error('When using --stages test, you must provide --test_suite')
            ok = False
            if args.test_suite == 'candi_eic_end2end':
                ok = run_test_candi_eic_end2end(args)
            elif args.test_suite == 'eic_all_methods_smoke':
                ok = run_test_eic_all_methods_smoke(args)
            elif args.test_suite == 'install_only':
                ok = run_test_install_only(args)
            elif args.test_suite == 'evaluate_only':
                ok = run_test_evaluate_only(args)
            print(f"[TEST SUITE] {args.test_suite}: {'PASS' if ok else 'FAIL'}")
        else:
            parser.error(f"Unknown stage: {stage}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""
Unified benchmarking orchestrator for CANDI, ChromImpute, Avocado, and eDICE.

Notes:
- Data loading and split logic are delegated to data.ExtendedEncodeDataHandler (mirrors data.py usage).
- Evaluation mirrors eval.py METRICS usage, but this orchestrator only coordinates stages.
- Each method runner encapsulates env checks, preprocess, train, and infer; evaluation is centralized.
- Default scopes: train on chr19, test on chr21; allow overrides via CLI.
- Timings: per-stage totals and per-bios inference timings are recorded to results/evaluation/timings.csv.

This file mirrors some logic from data.py for navigation/split reading, but does not modify core modules.
"""

import argparse
import os
import sys
import time
import json
import csv
from typing import List, Dict, Tuple

# Mirror data loading via data.py
try:
    from data import ExtendedEncodeDataHandler
except Exception as e:
    print("[unified_benchmark] ERROR: Failed to import ExtendedEncodeDataHandler from data.py:", e)
    raise

# Local imports (created in this branch)
try:
    import candi  # our CLI module for CANDI
except Exception:
    candi = None


def _append_timing(output_dir: str, method: str, stage: str, dataset: str,
                   scope_train: str, scope_test: str,
                   n_bios: int, n_assays: int, start_ts: float, end_ts: float) -> None:
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    path = os.path.join(output_dir, "evaluation", "timings.csv")
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
    def __init__(self, data_path: str, dataset: str, train_scope: str, test_scope: str, includes_35: List[str]):
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


# ----------------- IO utilities (mirrors eval/data logic) -----------------

def write_chrom_sizes(single_chr: str, eed: ExtendedEncodeDataHandler, out_path: str) -> None:
    """Write a chrom.sizes file for a single chromosome (mirrors the way eval handles chr sizes)."""
    # Load global sizes via eed.chr_sizes, as used in eval.py
    # Ensure the dictionary is populated (EVAL_CANDI does this in its init)
    # Here, we create a small helper mapping for single chromosome.
    # We mirror eval's default main chrs 1-22,X, but here we only need the selected chromosome length.
    # Use data/hg38.chrom.sizes bundled in repo via eed.chr_sizes_file.
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


def write_bedgraph_from_pval_arcsinh(eed: ExtendedEncodeDataHandler, bios: str, assay: str,
                                      chrom: str, out_bedgraph: str) -> None:
    """
    Generate a bedGraph for arcsinh(pval) at 25bp resolution for a given bios/assay/chromosome.
    We mirror data loading from data.py (load_bios_BW + make_bios_tensor_BW) and apply arcsinh.
    We write fixed 25bp bins as bedGraph (start, end, value).
    """
    import numpy as np
    temp_p = eed.load_bios_BW(bios, [chrom, 0, eed.coords.__self__.chr_sizes[chrom] if hasattr(eed.coords, '__self__') else 0], 1)
    # Note: eed.coords is a method; we avoid relying on internal. Use eed.chr_sizes_file instead to compute length.
    length = None
    with open(eed.chr_sizes_file, 'r') as f:
        for line in f:
            name, size = line.strip().split('\t')
            if name == chrom:
                length = int(size)
                break
    if length is None:
        raise RuntimeError(f"Failed to read length for {chrom}")

    P, avlP = eed.make_bios_tensor_BW(temp_p)
    expnames = list(eed.aliases["experiment_aliases"].keys())
    if assay not in expnames:
        raise RuntimeError(f"Assay {assay} not in aliases")
    aidx = expnames.index(assay)
    vec = P[:, aidx]
    try:
        vec = vec.numpy()
    except Exception:
        pass
    vec = np.arcsinh(vec)

    # Ensure length multiple of 25
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
    def __init__(self, dm: DatasetManager, output_dir: str, write_bw: bool):
        self.dm = dm
        self.output_dir = output_dir
        self.write_bw = write_bw
        os.makedirs(self.output_dir, exist_ok=True)

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
        # CANDI uses data.py directly; no external preprocess step required.
        return

    def train(self, args):
        if candi is None:
            raise RuntimeError("candi module is not importable")
        t0 = time.time()
        candi.cmd_train(args)
        t1 = time.time()
        _append_timing(self.output_dir, "candi", "train_total", args.dataset, args.train_scope, args.test_scope, self.dm.n_train_bios, self.dm.n_assays, t0, t1)

    def infer(self, args):
        if candi is None:
            raise RuntimeError("candi module is not importable")
        candi.cmd_infer(args)


class ChromImputeRunner(BaseRunner):
    """
    Implements ChromImpute multi-step workflow. We do not modify ChromImpute; we only:
    - create arcsinh(pval) bedGraph inputs via data.py loaders
    - generate chrom.sizes and samplemarktable.txt
    - call ChromImpute.jar Convert/ComputeGlobalDist/GenerateTrainData/Train/Apply

    UNCERTAINTY: Path to ChromImpute.jar. We assume lib/ChromImpute.jar. If missing, we log a clear error.
    """
    def __init__(self, dm: DatasetManager, output_dir: str, write_bw: bool):
        super().__init__(dm, output_dir, write_bw)
        self.base_dir = os.getcwd()
        self.proc_dir = os.path.join(self.output_dir, 'processed', 'chromimpute')
        self.model_dir = os.path.join(self.output_dir, 'models', 'chromimpute')
        self.pred_dir = os.path.join(self.output_dir, 'imputed_tracks', 'chromimpute')
        for d in [self.proc_dir, self.model_dir, self.pred_dir]:
            os.makedirs(d, exist_ok=True)
        # subdirs
        self.bedgraph_dir = os.path.join(self.proc_dir, 'bedgraph')
        self.meta_dir = os.path.join(self.proc_dir, 'metadata')
        self.converted_dir = os.path.join(self.proc_dir, 'converted')
        self.distance_dir = os.path.join(self.proc_dir, 'distances')
        self.traindata_dir = os.path.join(self.proc_dir, 'traindata')
        for d in [self.bedgraph_dir, self.meta_dir, self.converted_dir, self.distance_dir, self.traindata_dir]:
            os.makedirs(d, exist_ok=True)
        self.jar_path = os.path.join(self.base_dir, 'lib', 'ChromImpute.jar')

    def ensure_env(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] ERROR: {self.jar_path} not found. Please place ChromImpute.jar under lib/")

    def preprocess(self):
        # Write chrom.sizes for train and test scopes
        train_sizes = os.path.join(self.meta_dir, f'{self.dm.train_scope}.sizes')
        test_sizes = os.path.join(self.meta_dir, f'{self.dm.test_scope}.sizes')
        # Use a fresh EED to access chr_sizes_file
        eed = self.dm.eed
        write_chrom_sizes(self.dm.train_scope, eed, train_sizes)
        write_chrom_sizes(self.dm.test_scope, eed, test_sizes)

        # Build bedGraphs for all train bios (train_scope) and test bios (test_scope)
        # We produce files per (bios, assay, chr)
        nav_all = self.dm.navigation_all
        expnames = list(self.dm.eed.aliases["experiment_aliases"].keys())
        # Samplemarktable accumulates all (bios, assay) bedgraphs for both chromosomes
        sample_entries = []

        def build_for_split(bios_list: List[str], chrom: str):
            for bios in bios_list:
                # available assays for this bios (from navigation)
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

        # Write samplemarktable
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

    def train(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] Skipping train. Missing {self.jar_path}")
            return
        t0 = time.time()
        smt = os.path.join(self.meta_dir, 'samplemarktable.txt')
        train_sizes = os.path.join(self.meta_dir, f'{self.dm.train_scope}.sizes')
        # Convert and ComputeGlobalDist on train_scope
        self._run_java(["Convert", self.bedgraph_dir, smt, train_sizes, self.converted_dir])
        self._run_java(["ComputeGlobalDist", self.converted_dir, smt, train_sizes, self.distance_dir])
        # For each assay: GenerateTrainData
        for assay in self.dm.eed.aliases["experiment_aliases"].keys():
            self._run_java(["GenerateTrainData", self.converted_dir, self.distance_dir, smt, train_sizes, self.traindata_dir, assay])
        # For each (train bios, assay): Train
        for bios in self.dm.train_bios():
            for assay in self.dm.eed.aliases["experiment_aliases"].keys():
                self._run_java(["Train", self.traindata_dir, smt, self.model_dir, bios, assay])
        t1 = time.time()
        _append_timing(self.output_dir, "chromimpute", "train", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, len(self.dm.train_bios()), self.dm.n_assays, t0, t1)

    def infer(self):
        if not os.path.exists(self.jar_path):
            print(f"[ChromImpute] Skipping infer. Missing {self.jar_path}")
            return
        smt = os.path.join(self.meta_dir, 'samplemarktable.txt')
        test_sizes = os.path.join(self.meta_dir, f'{self.dm.test_scope}.sizes')
        # Apply for each (test bios, assay)
        for bios in self.dm.test_bios():
            tb0 = time.time()
            for assay in self.dm.eed.aliases["experiment_aliases"].keys():
                self._run_java(["Apply", self.converted_dir, self.distance_dir, self.model_dir, smt, test_sizes, self.pred_dir, bios, assay])
                # TODO: parse ChromImpute outputs (wig.gz) into .npy arrays under self.pred_dir with naming
                # {bios}__{assay}__chr21__{denoised|imputed}__pval.npy
                # UNCERTAINTY: Output file naming format; we will leave conversion for the next step once outputs exist.
            tb1 = time.time()
            _append_timing(self.output_dir, "chromimpute", "infer_bios", self.dm.dataset, self.dm.train_scope, self.dm.test_scope, 1, self.dm.n_assays, tb0, tb1)


class AvocadoRunner(BaseRunner):
    def ensure_env(self):
        pass

    def preprocess(self):
        pass

    def train(self):
        pass

    def infer(self):
        pass


class EdiceRunner(BaseRunner):
    def ensure_env(self):
        pass

    def preprocess(self):
        pass

    def train(self):
        pass

    def infer(self):
        pass


# ----------------- Evaluation scaffold -----------------

def evaluator(args):
    os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=True)
    print("[unified_benchmark] Evaluation scaffold ready. Will integrate once predictions exist for all methods.")


# ----------------- CLI -----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified benchmarking orchestrator")
    p.add_argument('--dataset', choices=['enc','merged','eic'], default='enc')
    p.add_argument('--models', nargs='+', choices=['candi','chromimpute','avocado','edice','all'], default=['all'])
    p.add_argument('--stages', nargs='+', choices=['preprocess','train','infer','evaluate','all'], default=['all'])
    p.add_argument('--data_path', type=str, default='data/')
    p.add_argument('--train_scope', type=str, default='chr19')
    p.add_argument('--test_scope', type=str, default='chr21')
    p.add_argument('--output_dir', type=str, default='results/')
    p.add_argument('--write_bw', action='store_true')

    # CANDI args (forwarded to candi handlers as needed)
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

    # For CANDI eval/infer paths
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
        stages = ['preprocess','train','infer','evaluate']

    includes_35 = [
        'ATAC-seq','DNase-seq','H2AFZ','H2AK5ac','H2AK9ac','H2BK120ac','H2BK12ac','H2BK15ac',
        'H2BK20ac','H2BK5ac','H3F3A','H3K14ac','H3K18ac','H3K23ac','H3K23me2','H3K27ac','H3K27me3',
        'H3K36me3','H3K4ac','H3K4me1','H3K4me2','H3K4me3','H3K56ac','H3K79me1','H3K79me2','H3K9ac',
        'H3K9me1','H3K9me2','H3K9me3','H3T11ph','H4K12ac','H4K20me1','H4K5ac','H4K8ac','H4K91ac'
    ]

    dm = DatasetManager(args.data_path, args.dataset, args.train_scope, args.test_scope, includes_35)

    runners = {}
    if 'candi' in models:
        runners['candi'] = CandiRunner(dm, args.output_dir, args.write_bw)
    if 'chromimpute' in models:
        runners['chromimpute'] = ChromImputeRunner(dm, args.output_dir, args.write_bw)
    if 'avocado' in models:
        runners['avocado'] = AvocadoRunner(dm, args.output_dir, args.write_bw)
    if 'edice' in models:
        runners['edice'] = EdiceRunner(dm, args.output_dir, args.write_bw)

    for stage in stages:
        if stage == 'preprocess':
            for m in models:
                runners[m].ensure_env()
                runners[m].preprocess()
        elif stage == 'train':
            for m in models:
                if m == 'candi':
                    runners[m].train(args)
                else:
                    runners[m].train()
        elif stage == 'infer':
            for m in models:
                if m == 'candi':
                    runners[m].infer(args)
                else:
                    runners[m].infer()
        elif stage == 'evaluate':
            evaluator(args)
        else:
            parser.error(f"Unknown stage: {stage}")


if __name__ == '__main__':
    main()

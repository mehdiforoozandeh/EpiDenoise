#!/usr/bin/env python3

"""
Unified benchmarking orchestrator for CANDI, ChromImpute, Avocado, and eDICE.

Notes:
- Data loading and split logic are delegated to data.ExtendedEncodeDataHandler (mirrors data.py usage).
- Evaluation mirrors eval.py METRICS usage, but this orchestrator only coordinates stages.
- Each method runner encapsulates env checks, preprocess, train, and infer; evaluation is centralized.
- Default scopes: train on chr19, test on chr21; allow overrides via CLI.
- Timings: per-stage totals and per-bios inference timings are recorded to results/evaluation/timings.csv.
"""

import argparse
import os
import sys
import time
from typing import List

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
    import csv
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
        # We use initialize_EED for training side to respect includes/merging/eic
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

    @property
    def n_train_bios(self) -> int:
        return len(self.eed.navigation)

    @property
    def n_assays(self) -> int:
        try:
            return len(self.eed.aliases["experiment_aliases"])
        except Exception:
            return 35


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
        # Call candi CLI function directly to avoid shell
        candi.cmd_train(args)
        t1 = time.time()
        _append_timing(self.output_dir, "candi", "train_total", args.dataset, args.train_scope, args.test_scope, self.dm.n_train_bios, self.dm.n_assays, t0, t1)

    def infer(self, args):
        if candi is None:
            raise RuntimeError("candi module is not importable")
        # candi runner handles per-bios timings internally
        candi.cmd_infer(args)


# Placeholders: will be implemented in subsequent steps with full logic
class ChromImputeRunner(BaseRunner):
    def ensure_env(self):
        # TODO: create/check conda env, install openjdk, download ChromImpute.jar into lib/
        pass

    def preprocess(self):
        # TODO: write arcsinh(pval) BedGraph for chr19 (train) and chr21 (apply)
        pass

    def train(self):
        # TODO: run Convert, ComputeGlobalDist, GenerateTrainData, Train
        pass

    def infer(self):
        # TODO: run Apply for all assays and test bios; save arrays; empirical variance; optional .bw
        pass


class AvocadoRunner(BaseRunner):
    def ensure_env(self):
        # TODO: create/check avocado_env and pip install avocado-epigenome
        pass

    def preprocess(self):
        # TODO: build per-(bios, assay) arcsinh(pval) caches (.npz) on chr19
        pass

    def train(self):
        # TODO: initialize Avocado with published defaults and fit on chr19
        pass

    def infer(self):
        # TODO: predict chr21 for all 35 assays; save denoised+imputed; empirical variance; optional .bw
        pass


class EdiceRunner(BaseRunner):
    def ensure_env(self):
        # TODO: create/check edice_env; clone repo; pip install -r requirements; setup.py install
        pass

    def preprocess(self):
        # TODO: build HDF5 data_matrix (chr19 arcsinh pval), idmap.json, splits.json
        pass

    def train(self):
        # TODO: call scripts/train_eDICE.py with arcsinh transform
        pass

    def infer(self):
        # TODO: parse integrated predictions; save arrays for denoised+imputed; empirical variance; optional .bw
        pass


def evaluator(args):
    """
    Central evaluation stage. We will implement after method runners are in place.
    For now, scaffold the function and create files.
    """
    os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=True)
    # Placeholder: will compute metrics using eval.py after predictions exist
    print("[unified_benchmark] Evaluation scaffold complete (metrics to be added after predictions exist).")


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

    # 35 assays from data.py initialize_EED default includes; we rely on data.py to hold the list.
    # UNCERTAINTY: data.py constant is embedded in function default; we pass an identical list here to keep clarity.
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

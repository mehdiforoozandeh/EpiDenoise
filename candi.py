#!/usr/bin/env python3

import argparse
import os
import time
import csv
import glob
from typing import Optional

# Import existing modules; we avoid modifying your core code
from data import ExtendedEncodeDataHandler
from train_candi import Train_CANDI, CANDI_LOADER
from eval import EVAL_CANDI, METRICS


def _auto_device_str() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _append_timing(output_dir: str, method: str, stage: str, dataset: str,
                   scope_train: Optional[str], scope_test: Optional[str],
                   n_bios: int, n_assays: int, start_ts: float, end_ts: float) -> None:
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    timings_path = os.path.join(output_dir, "evaluation", "timings.csv")
    row = {
        "method": method,
        "stage": stage,
        "dataset": dataset,
        "scope_train": scope_train or "",
        "scope_test": scope_test or "",
        "n_bios": n_bios,
        "n_assays": n_assays,
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "wall_time_sec": round(end_ts - start_ts, 3),
    }
    write_header = not os.path.exists(timings_path)
    with open(timings_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# -------- train --------

def cmd_train(args: argparse.Namespace) -> None:
    """
    Train CANDI using Train_CANDI from train_candi.py.
    Defaults: train_scope=chr19, 35 assays (from data.py), GPU if available.
    We only coordinate and record timing; core training logic remains unmodified.
    """
    # Build hyperparameters mirroring train_candi.py
    h = {
        "data_path": args.data_path,
        "dropout": args.dropout,
        "n_cnn_layers": args.n_cnn_layers,
        "conv_kernel_size": args.conv_kernel_size,
        "pool_size": args.pool_size,
        "expansion_factor": args.expansion_factor,
        "nhead": args.nhead,
        "n_sab_layers": args.n_sab_layers,
        "pos_enc": args.pos_enc,
        "epochs": args.epochs,
        "inner_epochs": args.inner_epochs,
        "mask_percentage": args.mask_percentage,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_loci": args.num_loci,
        "lr_halflife": args.lr_halflife,
        "min_avail": args.min_avail,
        "hpo": args.hpo,
        "separate_decoders": (not args.shared_decoders),
        "merge_ct": True,
        "loci_gen": args.train_scope,
        "must_have_chr_access": True,
        "optim": args.optim,
        "unet": args.unet,
        "LRschedule": args.LRschedule,
        "supertrack": args.supertrack,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    Train_CANDI(
        h,
        eic=(args.dataset == "eic"),
        checkpoint_path=args.from_ckpt,
        DNA=True,
        suffix=args.suffix,
        prog_mask=args.prog_mask,
        HPO=args.hpo,
    )
    t1 = time.time()

    # Estimate counts for timing row
    try:
        eed = ExtendedEncodeDataHandler(args.data_path, resolution=25)
        eed.initialize_EED(
            m=h["num_loci"],
            context_length=h["context_length"]*25,
            bios_batchsize=h["batch_size"],
            loci_batchsize=1,
            loci_gen=h["loci_gen"],
            bios_min_exp_avail_threshold=h["min_avail"],
            check_completeness=True,
            shuffle_bios=True,
            includes=eed.aliases["experiment_aliases"].keys() if hasattr(eed, "aliases") else [],
            excludes=[],
            merge_ct=True,
            must_have_chr_access=True,
            eic=(args.dataset == "eic"),
            DSF_list=[1,2],
        )
        n_bios = len(eed.navigation)
        n_assays = len(eed.aliases["experiment_aliases"]) if hasattr(eed, "aliases") else 35
    except Exception:
        n_bios, n_assays = -1, 35

    _append_timing(args.output_dir, "candi", "train", args.dataset, args.train_scope, None, n_bios, n_assays, t0, t1)


# -------- infer --------

def cmd_infer(args: argparse.Namespace) -> None:
    """
    Inference on test scope (default chr21). We mirror EVAL_CANDI.bios_pipeline with quick=False
    to obtain prediction arrays, and save only pval predictions (and std) for both
    denoised (upsampled) and imputed categories.

    IMPORTANT: We do not change eval.py; we only use its APIs. We ignore all count outputs.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    handler = EVAL_CANDI(
        model=args.ckpt,
        data_path=args.data_path,
        context_length=args.context_length,
        batch_size=args.batch_size,
        hyper_parameters_path=args.hyperparams,
        mode="dev",
        split="test",
        eic=(args.dataset == "eic"),
        DNA=True,
        savedir=os.path.join(args.output_dir, "evals"),
    )

    bios_list = list(handler.dataset.navigation.keys())

    t_total0 = time.time()
    for bios in bios_list:
        tb0 = time.time()
        try:
            res = handler.bios_pipeline(bios, x_dsf=args.dsf, quick=False)
            out_dir = os.path.join(args.output_dir, "imputed_tracks", "candi")
            os.makedirs(out_dir, exist_ok=True)
            for r in res:
                cmp = r.get("comparison", "None")
                if cmp not in ("imputed", "upsampled"):
                    continue
                category = "imputed" if cmp == "imputed" else "denoised"
                bios_name = r["bios"]
                assay = r["feature"]
                pval = r.get("pred_pval")
                pstd = r.get("pred_pval_std")
                if pval is None:
                    continue
                base = f"{bios_name}__{assay}__{args.test_scope}__{category}__pval"
                import numpy as np
                np.save(os.path.join(out_dir, base + ".npy"), pval)
                if pstd is not None:
                    np.save(os.path.join(out_dir, base.replace("__pval","__std") + ".npy"), pstd)
                # Optional .bw skipped unless a later flag requires it
        except Exception as e:
            print(f"[candi.infer] Failed {bios}: {e}")
        tb1 = time.time()
        _append_timing(args.output_dir, "candi", "infer_bios", args.dataset, None, args.test_scope, 1, -1, tb0, tb1)

    t_total1 = time.time()
    _append_timing(args.output_dir, "candi", "infer_total", args.dataset, None, args.test_scope, len(bios_list), -1, t_total0, t_total1)


# -------- eval --------

def cmd_eval(args: argparse.Namespace) -> None:
    """
    Evaluate saved CANDI predictions (pval-only). We do NOT compute/record any count metrics.
    For merged dataset, we mirror quick_eval_rnaseq via EVAL_CANDI.bios_rnaseq_eval(quick=True).
    """
    os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=True)
    summary_path = os.path.join(args.output_dir, "evaluation", "summary_metrics.csv")
    write_header = not os.path.exists(summary_path)

    handler = EVAL_CANDI(
        model=args.ckpt if args.ckpt else "",
        data_path=args.data_path,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode="dev",
        split="test",
        eic=(args.dataset == "eic"),
        DNA=True,
        savedir=os.path.join(args.output_dir, "evals"),
    )

    metrics = METRICS()
    pred_dir = os.path.join(args.output_dir, "imputed_tracks", "candi")
    files = glob.glob(os.path.join(pred_dir, f"*__{args.test_scope}__*__pval.npy"))

    rows = []
    for f in files:
        try:
            base = os.path.basename(f)  # bios__assay__chr21__{imputed|denoised}__pval.npy
            bios, assay, scope, category, _ = base.split("__")
            import numpy as np
            pred = np.load(f)
            # GT pval for chr21
            temp_p = handler.dataset.load_bios_BW(bios, [args.test_scope, 0, handler.chr_sizes[args.test_scope]], args.dsf)
            P, avlP = handler.dataset.make_bios_tensor_BW(temp_p)
            expnames = list(handler.dataset.aliases["experiment_aliases"].keys())
            if assay not in expnames:
                print(f"[candi.eval] Assay {assay} not in aliases; skipping")
                continue
            aidx = expnames.index(assay)
            try:
                gt = P[:, aidx].numpy()
            except Exception:
                gt = P[:, aidx]

            # PVAL-ONLY metrics
            row = {
                "dataset": args.dataset,
                "method": "candi",
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
            print(f"[candi.eval] Failed for {f}: {e}")

    if rows:
        with open(summary_path, "a", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    # RNA-seq (merged only)
    if args.rnaseq and args.dataset == "merged":
        print("[candi.eval] Running RNA-seq evaluation (quick)")
        out_dir = os.path.join(args.output_dir, "evaluation", "rnaseq")
        os.makedirs(out_dir, exist_ok=True)
        for bios in list(handler.dataset.navigation.keys()):
            try:
                res = handler.bios_rnaseq_eval(bios, x_dsf=args.dsf, quick=True)
                res.to_csv(os.path.join(out_dir, f"{bios}_rnaseq_quick.csv"))
            except Exception as e:
                print(f"[candi.eval] RNA-seq failed for {bios}: {e}")


# -------- CLI --------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CANDI: train / infer / eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument('--data_path', type=str, default='data/')
        sp.add_argument('--dataset', choices=['enc','merged','eic'], default='enc')
        sp.add_argument('--train_scope', type=str, default='chr19')
        sp.add_argument('--test_scope', type=str, default='chr21')
        sp.add_argument('--output_dir', type=str, default='results/')
        sp.add_argument('--dsf', type=int, default=1)
        sp.add_argument('--context_length', type=int, default=1200)
        sp.add_argument('--batch_size', type=int, default=25)
        sp.add_argument('--min_avail', type=int, default=3)
        sp.add_argument('--num_loci', type=int, default=5000)
        sp.add_argument('--suffix', type=str, default='')
        sp.add_argument('--from_ckpt', type=str, default=None)
        sp.add_argument('--hpo', action='store_true')
        sp.add_argument('--prog_mask', action='store_true')
        sp.add_argument('--supertrack', action='store_true')
        sp.add_argument('--optim', type=str, default='sgd')
        sp.add_argument('--unet', action='store_true')
        sp.add_argument('--LRschedule', type=str, default=None)
        sp.add_argument('--dropout', type=float, default=0.1)
        sp.add_argument('--n_cnn_layers', type=int, default=3)
        sp.add_argument('--conv_kernel_size', type=int, default=3)
        sp.add_argument('--pool_size', type=int, default=2)
        sp.add_argument('--expansion_factor', type=int, default=3)
        sp.add_argument('--nhead', type=int, default=9)
        sp.add_argument('--n_sab_layers', type=int, default=4)
        sp.add_argument('--pos_enc', type=str, default='relative')
        sp.add_argument('--epochs', type=int, default=10)
        sp.add_argument('--inner_epochs', type=int, default=1)
        sp.add_argument('--mask_percentage', type=float, default=0.2)
        sp.add_argument('--shared_decoders', action='store_true')
        sp.add_argument('--learning_rate', type=float, default=1e-3)

    # train
    pt = sub.add_parser('train', help='Train CANDI')
    add_shared(pt)

    # infer
    pi = sub.add_parser('infer', help='Run inference and save pval predictions')
    add_shared(pi)
    pi.add_argument('--ckpt', type=str, required=True)
    pi.add_argument('--hyperparams', type=str, required=True)

    # eval
    pe = sub.add_parser('eval', help='Evaluate saved pval predictions; optional RNA-seq (merged)')
    add_shared(pe)
    pe.add_argument('--ckpt', type=str, default=None)
    pe.add_argument('--rnaseq', action='store_true')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.time()

    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'infer':
        cmd_infer(args)
    elif args.cmd == 'eval':
        cmd_eval(args)
    else:
        parser.error('Unknown command')

    t1 = time.time()
    # High-level timing per command (optional)
    _append_timing(args.output_dir, "candi", f"cmd_{args.cmd}", args.dataset, args.train_scope, args.test_scope, -1, -1, t0, t1)


if __name__ == '__main__':
    main()

 
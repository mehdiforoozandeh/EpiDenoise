#!/usr/bin/env python3
"""
Diagnostic tool for inspecting how prompt metadata affects CANDI predictions.

Example:
    python diagnose_metadata.py \
        --model-dir models/20251103_135249_CANDI_merged_ccre_3000loci_Nov03 \
        --data-path ../DATA_CANDI_MERGED \
        --dataset merged \
        --bios-name H9_grp2_rep1 \
        --assay H3K4me3 \
        --task impute \
        --mut-depth 2 \
        --mut-read-length 10000 \
        --mut-platform "Illumina HiSeq 4000" \
        --mut-run-type paired-ended
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from pred import CANDIPredictor
from _utils import NegativeBinomial, Gaussian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose metadata utilisation in CANDI")
    parser.add_argument("--model-dir", required=True, type=str, help="Directory with config + checkpoint")
    parser.add_argument("--data-path", required=True, type=str, help="Base dataset path")
    parser.add_argument("--dataset", required=True, choices=["merged", "eic"], help="Dataset type")
    parser.add_argument("--bios-name", required=True, type=str, help="Biosample name")
    parser.add_argument("--assay", required=True, type=str, help="Assay name (matching experiment aliases)")
    parser.add_argument(
        "--task", default="impute", choices=["impute", "denoise"], help="Task type (affects masking logic)"
    )
    parser.add_argument(
        "--locus",
        nargs=3,
        default=["chr21", "0", "46709983"],
        help="Genomic locus: chrom start end (default chr21 full)",
    )
    parser.add_argument("--dsf", default=1, type=int, help="Downsampling factor (default: 1)")
    parser.add_argument("--y-prompt-spec", type=str, default=None, help="Optional JSON spec for prompt metadata")

    # Mutation options (applied to target assay)
    parser.add_argument("--mut-depth", type=float, default=None, help="Depth override (linear scale)")
    parser.add_argument("--mut-platform", type=str, default=None, help="Sequencing platform override")
    parser.add_argument("--mut-read-length", type=float, default=None, help="Read length override")
    parser.add_argument(
        "--mut-run-type",
        choices=["single-ended", "paired-ended"],
        default=None,
        help="Run type override",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Also evaluate with metadata masked to cloze token (-2) for sanity",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional limit on number of windows evaluated (for speed)",
    )
    return parser.parse_args()


@torch.no_grad()
def _metadata_embedding_stats(model: CANDIPredictor, metadata: torch.Tensor) -> Dict[str, float]:
    """Return simple stats for decoder-side metadata embeddings."""
    decoder = model.model.count_decoder if hasattr(model.model, "count_decoder") else model.model.decoder
    emb = decoder.ymd_emb(metadata.to(model.device))
    norms = torch.linalg.norm(emb, dim=-1)
    return {
        "mean_norm": float(norms.mean().item()),
        "std_norm": float(norms.std(unbiased=False).item()),
        "max_norm": float(norms.max().item()),
        "min_norm": float(norms.min().item()),
    }


def _weight_norms(module: torch.nn.Module) -> Dict[str, float]:
    stats = {}
    for name, param in module.named_parameters():
        stats[name] = float(param.detach().float().norm().item())
    return stats


def _collect_predictions(
    predictor: CANDIPredictor,
    X: torch.Tensor,
    mX: torch.Tensor,
    mY: torch.Tensor,
    avail: torch.Tensor,
    seq: torch.Tensor | None,
) -> Dict[str, np.ndarray]:
    n, p, mu, var, peak = predictor.predict(X, mX, mY, avail, seq)
    nb = NegativeBinomial(p, n)
    gaus = Gaussian(mu, var)
    pval_var = gaus.var
    if torch.is_tensor(pval_var):
        pval_var_arr = pval_var.cpu().numpy()
    else:
        pval_var_arr = np.asarray(pval_var)

    return {
        "count_mean": nb.mean().cpu().numpy(),
        "count_var": nb.var().cpu().numpy(),
        "pval_mean": gaus.mean().cpu().numpy(),
        "pval_var": pval_var_arr,
        "peak_logit": peak.cpu().numpy(),
        "p_param": p.cpu().numpy(),
        "n_param": n.cpu().numpy(),
    }


def _diff_summary(base: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    diff = other - base
    denom = np.maximum(np.abs(base), 1e-6)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_mae": float(np.mean(np.abs(diff) / denom)),
    }


def _compare_outputs(
    base: Dict[str, np.ndarray],
    other: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    return {key: _diff_summary(base[key], other[key]) for key in base.keys()}


def _limit_samples(
    tensors: Iterable[torch.Tensor],
    sample_count: int | None,
) -> Tuple[torch.Tensor, ...]:
    if sample_count is None:
        return tuple(tensors)
    limited = []
    for tensor in tensors:
        limited.append(tensor[:sample_count])
    return tuple(limited)


def main() -> None:
    args = parse_args()

    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]

    predictor = CANDIPredictor(args.model_dir)
    predictor.setup_data_handler(
        data_path=args.data_path,
        dataset_type=args.dataset,
        context_length=predictor.context_length,
        resolution=25,
        split="test",
    )

    if predictor.data_handler is None:
        raise RuntimeError("Failed to set up data handler")

    # Load base metadata specification if provided
    manual_spec = None
    if args.y_prompt_spec:
        manual_spec = json.loads(Path(args.y_prompt_spec).read_text())

    if args.assay not in predictor.data_handler.aliases["experiment_aliases"]:
        raise ValueError(f"Assay '{args.assay}' not found in experiment aliases.")
    assay_idx = list(predictor.data_handler.aliases["experiment_aliases"].keys()).index(args.assay)

    X, Y, P, seq, mX, mY, avX, avY = predictor.load_data(
        bios_name=args.bios_name,
        locus=locus,
        dsf=args.dsf,
        fill_y_prompt_spec=manual_spec,
        fill_prompt_mode="median" if manual_spec is None else "custom",
    )

    X, Y, P, seq, mX, mY, avX, avY = _limit_samples(
        (X, Y, P, seq, mX, mY, avX, avY), args.sample_count
    )

    base_mY = mY.clone()

    # Prepare mutation spec using helper to ensure consistent log handling
    mutation_spec: Dict[str, Dict[str, float | str]] = {}
    if any(
        value is not None
        for value in (args.mut_depth, args.mut_platform, args.mut_read_length, args.mut_run_type)
    ):
        mutation_spec[args.assay] = {}
        if args.mut_depth is not None:
            mutation_spec[args.assay]["depth"] = args.mut_depth
        if args.mut_platform is not None:
            mutation_spec[args.assay]["sequencing_platform"] = args.mut_platform
        if args.mut_read_length is not None:
            mutation_spec[args.assay]["read_length"] = args.mut_read_length
        if args.mut_run_type is not None:
            mutation_spec[args.assay]["run_type"] = args.mut_run_type

    # Build altered metadata tensors
    def expand_metadata(metadata_2d: torch.Tensor) -> torch.Tensor:
        if base_mY.ndim == 3:
            return metadata_2d.unsqueeze(0).repeat(base_mY.shape[0], 1, 1)
        return metadata_2d

    base_2d = base_mY[0].clone() if base_mY.ndim == 3 else base_mY.clone()
    mutated_mY = base_mY.clone()

    if mutation_spec:
        mutated_2d = predictor.data_handler.fill_in_prompt_manual(
            base_2d.clone(),
            mutation_spec,
            overwrite=True,
        )
        mutated_mY = expand_metadata(mutated_2d)

    missing_mY = base_mY.clone()
    missing_mY[:, :, assay_idx] = -1

    cloze_mY = base_mY.clone()
    if args.no_metadata:
        cloze_mY[:, :, assay_idx] = -2

    # Run predictions
    outputs = {}
    scenarios = {
        "baseline": base_mY,
        "mutated": mutated_mY,
        "missing": missing_mY,
    }
    if args.no_metadata:
        scenarios["cloze"] = cloze_mY

    for tag, metadata in scenarios.items():
        outputs[tag] = _collect_predictions(predictor, X, mX, metadata, avX, seq)

    comparisons = {tag: _compare_outputs(outputs["baseline"], out) for tag, out in outputs.items() if tag != "baseline"}

    # Embedding stats
    embedding_stats = {tag: _metadata_embedding_stats(predictor, meta) for tag, meta in scenarios.items()}

    metadata_module_norms = _weight_norms(
        predictor.model.count_decoder.ymd_emb if hasattr(predictor.model, "count_decoder") else predictor.model.decoder.ymd_emb
    )

    encoder_module_norms = _weight_norms(predictor.model.encoder.xmd_emb)

    print("=== Metadata Diagnostics ===")
    print(f"Biosample: {args.bios_name} | Assay: {args.assay} | Locus: {locus}")
    print(f"Windows evaluated: {X.shape[0]} | Context length: {X.shape[1]}")
    print()

    for tag, stats in embedding_stats.items():
        print(f"[{tag}] embedding norms: {stats}")

    print("\n=== Prediction deltas vs baseline ===")
    for tag, metrics in comparisons.items():
        print(f"\nScenario: {tag}")
        for key, values in metrics.items():
            print(f"  {key:>12s}: {values}")

    def _describe(arr: np.ndarray) -> Dict[str, float]:
        flat = arr.reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }

    print("\n=== Baseline prediction summary ===")
    for key, arr in outputs["baseline"].items():
        print(f"  {key:>12s}: {_describe(arr)}")

    print("\n=== Metadata embedding weight norms (decoder side) ===")
    for name, value in metadata_module_norms.items():
        print(f"  {name:>30s}: {value:.6f}")

    print("\n=== Metadata embedding weight norms (encoder side) ===")
    for name, value in encoder_module_norms.items():
        print(f"  {name:>30s}: {value:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


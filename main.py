"""Age estimation on APPA-REAL — DenseNet121 + ViT-B/16.

Entry point. Loads <model>.yaml, applies CLI overrides, snapshots the
post-override config + env + seed, then dispatches to train/eval.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

# Lazy / safe imports: keep argparse working even if torch isn't installed
# locally. The heavy stuff is imported inside main().


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Age estimation on APPA-REAL (DenseNet + ViT)"
    )
    parser.add_argument(
        "--model", choices=["densenet", "vit", "vit_baseline"], required=True,
        help="Which model pipeline to run.",
    )
    parser.add_argument(
        "--mode", choices=["train", "test", "all"], default="all",
        help="train = training only; test = run test set on best ckpt; all = both.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Override config YAML path. Default: configs/<model>.yaml",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data/appa-real-release",
        help="APPA-REAL extracted root directory.",
    )
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Mutate cfg in-place with CLI overrides (only when explicitly provided)."""
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    return cfg


def main() -> None:
    args = parse_args()

    # Heavy imports here so --help / argparse stays usable without torch.
    import torch

    from src.dataset import build_dataloaders
    from src.eval import evaluate_and_save_artifacts
    from src.models import build_model
    from src.train import run_training
    from src.utils import (
        build_criterion,
        ensure_dir,
        load_config,
        save_config_snapshot,
        save_env_snapshot,
        save_seed,
        set_seed,
        worker_init_fn,
    )

    # Resolve config path
    cfg_path = args.config or os.path.join("configs", f"{args.model}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    cfg = apply_cli_overrides(cfg, args)

    # Per-model output / ckpt dirs (fair experiment: same layout, different name)
    model_out_dir = ensure_dir(os.path.join(args.out_dir, args.model))
    ckpt_dir = ensure_dir(args.ckpt_dir)

    # Snapshot config / env / seed
    save_config_snapshot(cfg, os.path.join(model_out_dir, "config_snapshot.yaml"))
    save_env_snapshot(
        os.path.join(model_out_dir, "env_snapshot.txt"),
        extra={"model": args.model, "mode": args.mode, "config_path": cfg_path},
    )
    save_seed(int(cfg.get("seed", 42)), os.path.join(model_out_dir, "seed.txt"))

    # Determinism + RNG
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] model={args.model} | mode={args.mode} | device={device}")
    print(f"[main] config: {cfg_path}")
    print(f"[main] effective cfg: {cfg}")
    print(f"[main] out_dir: {model_out_dir} | ckpt_dir: {ckpt_dir}")

    # Data, model, criterion
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, data_root=args.data_root, worker_init_fn=worker_init_fn
    )
    model = build_model(args.model, pretrained=bool(cfg.get("pretrained", True)))
    criterion = build_criterion(cfg)

    best_ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    history_df = None

    if args.mode in ("train", "all"):
        best_val_mae, history_df, best_ckpt_path = run_training(
            model=model,
            model_name=args.model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=model_out_dir,
            ckpt_dir=ckpt_dir,
        )
        print(f"[main] training done. best val MAE = {best_val_mae:.4f}")

    if args.mode in ("test", "all"):
        # Rebuild a fresh model and load best checkpoint cleanly.
        eval_model = build_model(args.model, pretrained=False)
        results = evaluate_and_save_artifacts(
            model=eval_model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            model_name=args.model,
            out_dir=model_out_dir,
            ckpt_path=best_ckpt_path,
            history_df=history_df,
        )
        print(f"[main] test done. summary: {results}")


if __name__ == "__main__":
    main()

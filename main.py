import argparse
from pathlib import Path
from src import utils
from src import preprocessing, train as train_mod, evaluate as evaluate_mod, interpret


def _merge_configs(default_cfg_path: str, override_cfg_path: str | None):
    cfg = utils.load_config(default_cfg_path)
    if override_cfg_path:
        cfg_override = utils.load_config(override_cfg_path)
        utils.deep_update(cfg, cfg_override)
    utils.ensure_project_dirs(cfg)
    utils.set_seed(cfg["project"]["random_seed"])
    return cfg


def cli():
    p = argparse.ArgumentParser(description="EWS-ML pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # preprocess
    sp_prep = sub.add_parser("preprocess", help="Build processed tables from raw or synthetic data")
    sp_prep.add_argument("--config", default="configs/default.yaml")

    # train
    sp_train = sub.add_parser("train", help="Train a model end-to-end")
    sp_train.add_argument("--config", default="configs/default.yaml")
    sp_train.add_argument("--model-config", default="configs/model_gbm.yaml")

    # evaluate
    sp_eval = sub.add_parser("evaluate", help="Evaluate a trained model on held-out test set")
    sp_eval.add_argument("--config", default="configs/default.yaml")
    sp_eval.add_argument("--model-path", default="checkpoints/model.pkl")

    # interpret
    sp_interp = sub.add_parser("interpret", help="Compute feature importance/attributions")
    sp_interp.add_argument("--config", default="configs/default.yaml")
    sp_interp.add_argument("--model-path", default="checkpoints/model.pkl")

    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    if args.cmd == "preprocess":
        cfg = _merge_configs(args.config, None)
        preprocessing.run(cfg)

    elif args.cmd == "train":
        cfg = _merge_configs(args.config, args.model_config)
        train_mod.run(cfg)

    elif args.cmd == "evaluate":
        cfg = _merge_configs(args.config, None)
        evaluate_mod.run(cfg, model_path=args.model_path)

    elif args.cmd == "interpret":
        cfg = _merge_configs(args.config, None)
        interpret.run(cfg, model_path=args.model_path)

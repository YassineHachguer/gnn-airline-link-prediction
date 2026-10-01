"""
Lance une expérience sur tous les splits et ajoute les résultats à results/runs.csv.

    python src/run.py --config configs/gae_gcn.yaml
    python src/run.py --config configs/gae_gcn.yaml --eval val   # mise au point (validation seulement)
    python src/run.py --method distance                          # méthode sans paramètres
    python src/run.py --summary                                  # tableau moyenne ± écart-type

Un fichier de config = une expérience :
    name: gae_gcn          # nom de l'expérience dans runs.csv
    method: gae            # méthode (voir methods/__init__.py)
    params: {...}          # paramètres passés à la méthode

Une ligne de runs.csv = une méthode × un split × un ensemble (val/test) × un type de négatifs.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from evaluate import evaluate
from features import load_metadata
from methods import METHODS
from split import load_split


DEFAULT_SPLITS = Path("data/splits")
DEFAULT_METADATA = Path("data/clean/nodes_metadata.csv")
DEFAULT_RESULTS = Path("results/runs.csv")


def run(exp_name, method_name, config, eval_sets, splits_dir, metadata_path, results_path):
    meta = load_metadata(metadata_path)
    rows = []

    for split_path in sorted(splits_dir.glob("test*_seed*.npz")):
        scenario, seed = split_path.stem.split("_")      # ex. "test10", "seed0"
        split = load_split(split_path)

        method = METHODS[method_name](**config).fit(split, meta)

        for eval_set in eval_sets:                        # "val" et/ou "test"
            pos_scores = method.score(split[f"{eval_set}_pos"])
            for negatives in ("random", "hard"):
                neg_scores = method.score(split[f"{eval_set}_neg_{negatives}"])
                rows.append({
                    "method": exp_name,
                    "config": json.dumps(config, sort_keys=True),
                    "scenario": scenario,
                    "seed": int(seed.replace("seed", "")),
                    "eval_set": eval_set,
                    "negatives": negatives,
                    **evaluate(pos_scores, neg_scores),
                    "date": datetime.now().isoformat(timespec="seconds"),
                })
        print(f"{exp_name} | {split_path.stem} : OK")

    df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
    print(f"\n{len(df)} lignes ajoutées à {results_path}")


def summary(results_path, eval_set="test"):
    df = pd.read_csv(results_path)
    df = df[df["eval_set"] == eval_set]
    # Si une méthode a été relancée, on garde le run le plus récent pour chaque split
    df = df.sort_values("date").drop_duplicates(
        ["method", "config", "scenario", "seed", "eval_set", "negatives"], keep="last")

    stats = df.groupby(["method", "negatives", "scenario"])[["auc", "ap", "hits@50"]].agg(["mean", "std"])
    table = pd.DataFrame(index=stats.index)
    for metric in ("auc", "ap", "hits@50"):
        table[metric] = (stats[(metric, "mean")].map("{:.3f}".format) + " ± "
                         + stats[(metric, "std")].map("{:.3f}".format))
    print(table.unstack("scenario").to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--config", type=Path, help="fichier YAML de l'expérience (configs/*.yaml)")
    parser.add_argument("--eval", default="val,test", help="val, test ou val,test")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        summary(args.results)
    elif args.config:
        exp = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        run(exp.get("name", args.config.stem), exp["method"], exp.get("params") or {},
            args.eval.split(","), args.splits, args.metadata, args.results)
    elif args.method:
        run(args.method, args.method, {}, args.eval.split(","),
            args.splits, args.metadata, args.results)
    else:
        parser.error("préciser --config, --method ou --summary")


if __name__ == "__main__":
    main()
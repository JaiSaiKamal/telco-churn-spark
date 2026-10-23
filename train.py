"""Train Apache Spark MLlib classification models from the Kaggle CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import MODEL_CONFIGS, train

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    parser.add_argument("--model", default=str(ROOT / "models"))
    parser.add_argument(
        "--model-type",
        choices=["all"] + list(MODEL_CONFIGS.keys()),
        default="all",
        help="Model algorithm to train. Default 'all' trains Logistic Regression, Random Forest, GBT, and Decision Tree.",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Download the CSV from Kaggle and place it at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv. "
            "See data/raw/README.md."
        )

    print(f"Starting Spark MLlib training (model type: {args.model_type})...")
    metrics = train(args.data, args.model, model_types=args.model_type)

    # Save metrics JSON to the models directory
    model_dir = Path(args.model)
    if model_dir.name == "telco_churn_pipeline":
        model_dir = model_dir.parent
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("                SPARK MLlib MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print("-" * 65)
    for model_name, m in metrics.get("models", {}).items():
        is_best = " (Best)" if model_name == metrics.get("best_model") else ""
        print(f"{model_name + is_best:<30} {m['accuracy']:<10.3f} {m['f1']:<10.3f} {m['roc_auc']:<10.3f}")
    print("=" * 65)
    print(f"Best Model: {metrics.get('best_model')}")
    print(f"Total Rows: {metrics.get('rows'):,} | Train: {metrics.get('train_rows'):,} | Test: {metrics.get('test_rows'):,}")
    print(f"Metrics saved to: {metrics_path}")


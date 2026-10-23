"""Spark MLlib training, evaluation, and inference for Telco churn."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os

# Configure JAVA_HOME on Linux / Streamlit Cloud if not already set
if "JAVA_HOME" not in os.environ and Path("/usr/lib/jvm/default-java").exists():
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"

# Configure HADOOP_HOME on Windows if not already set
if "HADOOP_HOME" not in os.environ:
    if Path("C:/hadoop/bin/winutils.exe").exists():
        os.environ["HADOOP_HOME"] = "C:\\hadoop"
        os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ.get("PATH", "")
    elif (Path(__file__).resolve().parent.parent / "hadoop/bin/winutils.exe").exists():
        hdir = Path(__file__).resolve().parent.parent / "hadoop"
        os.environ["HADOOP_HOME"] = str(hdir)
        os.environ["PATH"] = str(hdir / "bin") + ";" + os.environ.get("PATH", "")

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
LABEL = "Churn"

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "desc": "Linear classification baseline with L2 regularization",
    },
    "random_forest": {
        "name": "Random Forest",
        "desc": "Ensemble of 50 decision trees with feature bagging",
    },
    "gbt": {
        "name": "Gradient-Boosted Trees (GBT)",
        "desc": "Sequential boosting optimizing pseudo-residuals",
    },
    "decision_tree": {
        "name": "Decision Tree",
        "desc": "Single tree classifier for rule-based interpretability",
    },
}


def make_spark(app_name: str = "TelcoChurn") -> SparkSession:
    session = (SparkSession.builder.master("local[2]").appName(app_name)
               .config("spark.ui.enabled", "false")
               .config("spark.driver.bindAddress", "127.0.0.1")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.driver.memory", "512m")
               # Use algorithm v1 to avoid NativeIO filesystem commits on Windows
               .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1")
               .getOrCreate())

    # Disable Hadoop native IO (avoids UnsatisfiedLinkError for NativeIO$Windows.access0
    # when hadoop.dll version does not match the Hadoop jars bundled inside PySpark).
    hconf = session.sparkContext._jsc.hadoopConfiguration()
    hconf.setBoolean("io.native.lib.available", False)
    hconf.set("fs.file.impl.disable.cache", "true")

    return session


def load_dataset(spark: SparkSession, csv_path: str | Path) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", False).csv(str(csv_path))
    missing = sorted(set(FEATURES + [LABEL]) - set(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    # TotalCharges contains whitespace for a few zero-tenure customers.
    df = df.withColumn("TotalCharges", F.when(F.trim("TotalCharges") == "", None).otherwise(F.trim("TotalCharges")).cast("double"))
    for col in NUMERIC:
        df = df.withColumn(col, F.col(col).cast("double"))
    for col in CATEGORICAL + [LABEL]:
        df = df.withColumn(col, F.trim(F.col(col).cast("string")))
    df = df.na.fill(0.0, subset=NUMERIC).na.fill("Unknown", subset=CATEGORICAL)
    df = df.filter(F.col(LABEL).isin("Yes", "No"))
    return df


def get_classifier(model_type: str):
    """Instantiate a Spark MLlib classifier according to the requested model type."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.05,
            elasticNetParam=0.0,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=5,
            seed=42,
        )
    elif model_type == "gbt":
        return GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            maxDepth=4,
            seed=42,
        )
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            seed=42,
        )
    else:
        valid = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {valid}")


def build_pipeline(model_type: str = "logistic_regression") -> Pipeline:
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep") for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"{c}__idx" for c in CATEGORICAL],
                            outputCols=[f"{c}__ohe" for c in CATEGORICAL], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=NUMERIC + [f"{c}__ohe" for c in CATEGORICAL],
                                outputCol="features", handleInvalid="keep")
    label_indexer = StringIndexer(inputCol=LABEL, outputCol="label", stringOrderType="alphabetAsc", handleInvalid="error")
    classifier = get_classifier(model_type)
    return Pipeline(stages=indexers + [encoder, assembler, label_indexer, classifier])


def extract_feature_importances(pipeline_model: PipelineModel, transformed_df: DataFrame) -> list[dict[str, Any]]:
    """Extract top feature importances or coefficient magnitudes from the fitted pipeline model."""
    try:
        classifier = pipeline_model.stages[-1]
        scores: list[float] | None = None
        if hasattr(classifier, "featureImportances"):
            scores = classifier.featureImportances.toArray().tolist()
        elif hasattr(classifier, "coefficients"):
            scores = [abs(float(c)) for c in classifier.coefficients.toArray()]

        if not scores:
            return []

        # Retrieve feature names from metadata produced by VectorAssembler/OneHotEncoder
        schema_field = transformed_df.schema["features"]
        attrs = schema_field.metadata.get("ml_attr", {}).get("attrs", {})
        all_attrs = []
        for kind in ("numeric", "binary", "nominal"):
            all_attrs.extend(attrs.get(kind, []))
        all_attrs.sort(key=lambda x: x.get("idx", 0))

        if len(all_attrs) == len(scores):
            names = [a.get("name", f"feat_{i}") for i, a in enumerate(all_attrs)]
        else:
            names = [f"Feature {i+1}" for i in range(len(scores))]

        # Simplify names for cleaner UI presentation
        clean_names = [n.replace("__ohe", "").replace("__idx", "") for n in names]
        paired = [{"feature": n, "importance": round(float(s), 4)} for n, s in zip(clean_names, scores)]
        paired.sort(key=lambda x: x["importance"], reverse=True)
        return paired
    except Exception:
        return []


def train(csv_path: str | Path, model_dir_or_path: str | Path, model_types: list[str] | str = "all") -> dict[str, Any]:
    """Train one or multiple Spark MLlib classification models, evaluate on held-out split, and save models."""
    spark = make_spark("TelcoChurnTraining")
    base_dir = Path(model_dir_or_path)
    # If the user passed a path directly to 'telco_churn_pipeline', use its parent folder
    if base_dir.name == "telco_churn_pipeline":
        base_dir = base_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    if model_types == "all":
        types_to_train = list(MODEL_CONFIGS.keys())
    elif isinstance(model_types, str):
        types_to_train = [model_types]
    else:
        types_to_train = list(model_types)

    try:
        data = load_dataset(spark, csv_path).cache()
        total = data.count()
        if total < 20:
            raise ValueError(f"Dataset has only {total} rows; at least 20 are needed.")

        churn_count = data.where(F.col(LABEL) == "Yes").count()
        churn_rate = churn_count / total

        train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.cache()
        test_df = test_df.cache()

        train_count = train_df.count()
        test_count = test_df.count()

        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

        models_results: dict[str, Any] = {}
        best_auc = -1.0
        best_model_key = types_to_train[0]
        best_pipeline_model: PipelineModel | None = None

        for m_key in types_to_train:
            m_info = MODEL_CONFIGS[m_key]
            pipeline = build_pipeline(model_type=m_key)
            fitted_model = pipeline.fit(train_df)
            predictions = fitted_model.transform(test_df).cache()

            acc = float(acc_eval.evaluate(predictions))
            f1 = float(f1_eval.evaluate(predictions))
            auc = float(auc_eval.evaluate(predictions))

            feature_imp = extract_feature_importances(fitted_model, predictions)

            save_path = base_dir / m_key
            fitted_model.write().overwrite().save(str(save_path))

            models_results[m_info["name"]] = {
                "key": m_key,
                "name": m_info["name"],
                "description": m_info["desc"],
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
                "path": str(save_path),
                "top_features": feature_imp[:10],
            }

            if auc > best_auc:
                best_auc = auc
                best_model_key = m_key
                best_pipeline_model = fitted_model

            predictions.unpersist()

        # Save the best model also under 'telco_churn_pipeline' for backward compatibility
        if best_pipeline_model is not None:
            best_compat_path = base_dir / "telco_churn_pipeline"
            best_pipeline_model.write().overwrite().save(str(best_compat_path))

        best_model_name = MODEL_CONFIGS[best_model_key]["name"]
        best_metrics = models_results[best_model_name]

        combined_metrics = {
            "rows": total,
            "train_rows": train_count,
            "test_rows": test_count,
            "churn_rate": round(churn_rate, 4),
            "best_model": best_model_name,
            "best_model_key": best_model_key,
            "models": models_results,
            # Top-level metrics reflecting the best model for backward compatibility
            "accuracy": best_metrics["accuracy"],
            "f1": best_metrics["f1"],
            "roc_auc": best_metrics["roc_auc"],
        }

        train_df.unpersist()
        test_df.unpersist()
        data.unpersist()

        return combined_metrics
    finally:
        spark.stop()


def load_model(spark: SparkSession, model_path: str | Path) -> PipelineModel:
    return PipelineModel.load(str(model_path))


def predict_records(spark: SparkSession, model: PipelineModel, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    rows = []
    for record in records:
        row = {col: record.get(col) for col in FEATURES}
        # The fitted pipeline contains a label StringIndexer. Supply a valid
        # placeholder during inference; the classifiers ignore this output.
        row[LABEL] = "No"
        row["TotalCharges"] = float(row.get("TotalCharges") or 0)
        row["MonthlyCharges"] = float(row.get("MonthlyCharges") or 0)
        row["tenure"] = float(row.get("tenure") or 0)
        row["SeniorCitizen"] = float(row.get("SeniorCitizen") or 0)
        rows.append(row)
    result = model.transform(spark.createDataFrame(rows)).select("probability", "prediction").collect()
    return [{"churn_probability": float(r.probability[1]), "prediction": "Yes" if int(r.prediction) == 1 else "No"} for r in result]

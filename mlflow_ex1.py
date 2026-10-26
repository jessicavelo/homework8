# pip install mlflow pyspark
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, NaiveBayes

# ── SETUP ─────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Jessica")
print("Experiment ID:", mlflow.get_experiment_by_name("Jessica").experiment_id)

# ── DATASET ───────────────────────────────────────────────────────────────────
# Using a small inline dataset to avoid needing to download IMDB.
# In production, replace with: datasets.load_dataset("imdb") via HuggingFace.

'''reviews = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Brilliant performances and a gripping storyline. Highly recommend.",
    "One of the best films I have ever seen. A masterpiece.",
    "Great direction and superb acting. Will watch again.",
    "Incredible movie with a beautiful ending. Loved it.",
    "Terrible film. Boring, slow, and a complete waste of time.",
    "Awful acting and a nonsensical plot. I want my money back.",
    "One of the worst movies ever made. Do not watch.",
    "Dull and predictable. Fell asleep halfway through.",
    "Disappointing in every way. The script was a disaster.",
]'''

#labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

'''X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42
)'''

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("imdb-sentiment").getOrCreate()

ds = load_dataset("stanfordnlp/imdb")["train"].shuffle(seed=42).select(range(4500))
pdf = ds.to_pandas()
df = spark.createDataFrame(pdf)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Total: {df.count()}")
print(f"Train: {train_df.count()}")
print(f"Test: {test_df.count()}")
print("\nLabel distribution (train):")
train_df.groupBy("label").count().show()


# ── CANDIDATE MODELS ──────────────────────────────────────────────────────────
# Each pipeline = TF-IDF vectorizer + classifier.
# TF-IDF converts raw text into numerical feature vectors.
# MLflow records every combination so we can compare them in the UI.
def make_pipeline(num_features, classifier):
    return Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        HashingTF(inputCol="words", outputCol="tf", numFeatures=num_features),
        IDF(inputCol="tf", outputCol="features"),
        classifier,
    ])

candidates = [
    {
        "run_name": "tfidf-logistic",
        "pipeline": make_pipeline(5000, LogisticRegression(maxIter=100)),
        "params": {"vectorizer": "tfidf", "model": "LogisticRegression", "max_features": 5000},
    },
    {
        "run_name": "tfidf-linearsvc",
        "pipeline": make_pipeline(5000, LinearSVC(maxIter=100)),
        "params": {"vectorizer": "tfidf", "model": "LinearSVC", "max_features": 5000},
    },
    {
        "run_name": "tfidf-randomforest",
        "pipeline": make_pipeline(3000, RandomForestClassifier(numTrees=100, seed=42)),
        "params": {"vectorizer": "tfidf", "model": "RandomForest", "max_features": 3000},
    },
    {
        "run_name": "tfidf-naivebayes",
        "pipeline": make_pipeline(5000, NaiveBayes()),
        "params": {"vectorizer": "tfidf", "model": "NaiveBayes", "max_features": 5000},
    },
]

# ── RUN ALL CANDIDATES ────────────────────────────────────────────────────────
run_results = []

for candidate in candidates:
    with mlflow.start_run(run_name=candidate["run_name"]) as run:

        # Train Spark ML pipeline
        model = candidate["pipeline"].fit(train_df)
        preds = model.transform(test_df)

        # Convert to pandas to compute metrics
        preds_pdf = preds.select("label", "prediction").toPandas()

        metrics = {
            "accuracy": accuracy_score(preds_pdf["label"], preds_pdf["prediction"]),
            "f1_score": f1_score(preds_pdf["label"], preds_pdf["prediction"], zero_division=0),
        }

        mlflow.log_params(candidate["params"])
        mlflow.log_metrics(metrics)

        # Log the Spark ML model
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            registered_model_name="imdb-sentiment"
        )

        run_results.append({
            "run_id"  : run.info.run_id,
            "run_name": candidate["run_name"],
            **metrics
        })
        print(f"{candidate['run_name']:25s} | acc={metrics['accuracy']:.4f} | "
              f"f1={metrics['f1_score']:.4f}")

# ── FIND BEST & ASSIGN ALIAS ──────────────────────────────────────────────────
best = max(run_results, key=lambda x: x["f1_score"])
print(f"\nBest model: {best['run_name']} (F1={best['f1_score']:.4f})")

client = MlflowClient()

# Get the registered model version for the best run
versions = client.search_model_versions("name='imdb-sentiment'")
best_version = next(v.version for v in versions if v.run_id == best["run_id"])

# Assign champion alias
client.set_registered_model_alias("imdb-sentiment", "champion", version=best_version)

print(f"✅ Best model: {best['run_name']} (F1={best['f1_score']:.4f})")
print(f"   Run ID: {best['run_id']}")
print(f"   Version: {best_version}")
print(f"   Alias: 'champion' ✓")
print("\n✅ Steps 1-6 Complete! Ready for Step 7.")

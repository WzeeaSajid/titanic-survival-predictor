import os
import json
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.pipeline import (
    TitanicFeatureEngineer,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

load_dotenv()

# ============================================================
# PATHS
# ============================================================
DATA_PATH  = os.getenv("INPUT_FILE", "data/raw/train.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
META_PATH  = os.getenv("META_PATH",  "models/model_meta.json")
SUBMISSION_FEATURES_PATH = "models/feature_columns.json"

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1/7] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ============================================================
# STEP 2 — EDA REPORT
# ============================================================
def run_eda(df: pd.DataFrame):
    print("\n[2/7] EDA Report")
    print("=" * 60)

    print(f"Shape        : {df.shape}")
    print(f"Duplicates   : {df.duplicated().sum()}")

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in missing[missing > 0].index:
        print(f"  {col:<15}: {missing[col]:>4} missing ({missing_pct[col]}%)")

    print("\n--- Target Distribution ---")
    counts = df["Survived"].value_counts()
    print(f"  Not Survived (0): {counts[0]}  ({counts[0]/len(df)*100:.1f}%)")
    print(f"  Survived     (1): {counts[1]}  ({counts[1]/len(df)*100:.1f}%)")

    print("\n--- Numeric Summary ---")
    print(df[["Age", "Fare", "SibSp", "Parch"]].describe().round(2))

    print("\n--- Categorical Summary ---")
    for col in ["Sex", "Embarked", "Pclass"]:
        print(f"  {col}: {df[col].value_counts().to_dict()}")

    print("=" * 60)


# ============================================================
# STEP 3 — BUILD SKLEARN PIPELINE
# ============================================================
def build_pipeline() -> Pipeline:
    print("\n[3/7] Building sklearn pipeline")

    # Numeric: impute missing with median
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical: impute missing with most_frequent, then encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # Combine both transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop everything not listed — clean and explicit
    )

    # Full pipeline: engineer → preprocess → model
    pipeline = Pipeline(steps=[
        ("feature_engineer", TitanicFeatureEngineer()),
        ("preprocessor",     preprocessor),
        ("classifier",       RandomForestClassifier(
            random_state=42,
            class_weight="balanced",  # handles 61/39 imbalance
            n_jobs=-1,
        )),
    ])

    print("      Pipeline built successfully")
    return pipeline


# ============================================================
# STEP 4 — TRAIN / VALIDATION SPLIT
# ============================================================
def split_data(df: pd.DataFrame):
    print("\n[4/7] Splitting data")

    TARGET = "Survived"
    DROP_COLS = ["PassengerId", "Survived"]

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # preserve 61/39 ratio in both splits
    )

    print(f"      Train : {X_train.shape[0]} rows")
    print(f"      Val   : {X_val.shape[0]} rows")
    return X_train, X_val, y_train, y_val


# ============================================================
# STEP 5 — HYPERPARAMETER TUNING
# ============================================================
def tune_pipeline(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    print("\n[5/7] Hyperparameter tuning (RandomizedSearchCV)")

    param_dist = {
        "classifier__n_estimators":      [100, 200, 300, 500],
        "classifier__max_depth":         [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 2, 4],
        "classifier__max_features":      ["sqrt", "log2"],
        "classifier__bootstrap":         [True, False],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,           # 50 random combinations
        scoring="accuracy",  # Kaggle scores on accuracy
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit=True,          # refit best model on full X_train after search
    )

    search.fit(X_train, y_train)

    print(f"\n      Best CV Accuracy : {search.best_score_:.4f}")
    print(f"      Best Params      : {search.best_params_}")

    return search.best_estimator_


# ============================================================
# STEP 6 — EVALUATE
# ============================================================
def evaluate(pipeline: Pipeline, X_val, y_val, X_train, y_train):
    print("\n[6/7] Evaluation")
    print("=" * 60)

    y_pred      = pipeline.predict(X_val)
    y_pred_prob = pipeline.predict_proba(X_val)[:, 1]

    acc       = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall    = recall_score(y_val, y_pred)
    f1        = f1_score(y_val, y_pred)
    roc_auc   = roc_auc_score(y_val, y_pred_prob)
    cm        = confusion_matrix(y_val, y_pred)

    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_val, y_pred)}")

    # Cross-validation on training data
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"\n  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("=" * 60)

    return {
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
        "cv_mean":   round(cv_scores.mean(), 4),
        "cv_std":    round(cv_scores.std(), 4),
    }


# ============================================================
# STEP 7 — SAVE MODEL + METADATA
# ============================================================
def save_artifacts(pipeline: Pipeline, metrics: dict, best_params: dict):
    print(f"\n[7/7] Saving artifacts")

    # Save the full pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"      Model saved   : {MODEL_PATH}")

    # Save feature columns (contract between train and inference)
    feature_columns = {
        "numeric":     NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    }
    with open(SUBMISSION_FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"      Features saved: {SUBMISSION_FEATURES_PATH}")

    # Save metadata (for audit trail)
    meta = {
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "model_path":  MODEL_PATH,
        "metrics":     metrics,
        "best_params": best_params,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      Meta saved    : {META_PATH}")


# ============================================================
# KAGGLE SUBMISSION GENERATOR
# ============================================================
def generate_submission(pipeline: Pipeline):
    test_path = "data/raw/test.csv"
    if not os.path.exists(test_path):
        print("\n[SKIP] test.csv not found — skipping submission generation")
        return

    print("\n[BONUS] Generating Kaggle submission")
    test_df = pd.read_csv(test_path)

    passenger_ids = test_df["PassengerId"]
    X_test = test_df.drop(columns=["PassengerId"])

    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived":    predictions,
    })

    out_path = "submission/submission.csv"
    os.makedirs("submission", exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"      Submission saved: {out_path}")
    print(f"      Rows: {len(submission)}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TITANIC — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    run_eda(df)

    # 3. Build pipeline
    pipeline = build_pipeline()

    # 4. Split
    X_train, X_val, y_train, y_val = split_data(df)

    # 5. Tune
    best_pipeline = tune_pipeline(pipeline, X_train, y_train)

    # 6. Evaluate
    metrics = evaluate(best_pipeline, X_val, y_val, X_train, y_train)

    # 7. Save
    # Extract best params for metadata
    rf_params = best_pipeline.named_steps["classifier"].get_params()
    save_artifacts(best_pipeline, metrics, rf_params)

    # Bonus — Kaggle submission
    generate_submission(best_pipeline)

    print("\nTraining complete\n")

import joblib
import pickle
import pandas as pd
import numpy as np
import os
import logging

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# MODEL — loaded ONCE at module level
# Never inside the predict function
# This means the model loads when the API starts, not per request
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

try:
    _pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from: {MODEL_PATH}")
except (FileNotFoundError, EOFError, pickle.UnpicklingError, ModuleNotFoundError) as e:
    _pipeline = None
    logger.error(
        f"Failed to load model from {MODEL_PATH}: {e}. "
        "Run src/train.py first."
    )


# ============================================================
# INTERPRETATION HELPER
# ============================================================
def _interpret(probability: float) -> str:
    if probability >= 0.75:
        return "Very likely to survive"
    elif probability >= 0.55:
        return "Likely to survive"
    elif probability >= 0.45:
        return "Uncertain"
    elif probability >= 0.25:
        return "Unlikely to survive"
    else:
        return "Very unlikely to survive"


# ============================================================
# MAIN PREDICT FUNCTION
# ============================================================
def predict_survival(
    pclass: int,
    sex: str,
    age: float,
    fare: float,
    sibsp: int,
    parch: int,
    embarked: str,
    cabin: str = "UNKNOWN",
    name: str = "Unknown, Mr. Unknown",
    ticket: str = "NONE",
) -> dict:
    """
    Predict survival for a single passenger.

    Parameters
    ----------
    pclass   : Passenger class (1, 2, or 3)
    sex      : 'male' or 'female'
    age      : Age in years
    fare     : Ticket fare paid
    sibsp    : Number of siblings/spouses aboard
    parch    : Number of parents/children aboard
    embarked : Port of embarkation ('C', 'Q', or 'S')
    cabin    : Cabin number (optional, defaults to UNKNOWN)
    name     : Passenger name (optional, used for Title extraction)
    ticket   : Ticket number (optional, used for TicketPrefix)

    Returns
    -------
    dict with keys:
        survived      : 0 or 1
        probability   : float between 0 and 1
        interpretation: human-readable string
    """

    if _pipeline is None:
        raise RuntimeError(
            "Model is not loaded. Run src/train.py to generate model.pkl first."
        )

    # ── Build a single-row DataFrame ─────────────────────────
    # Column names must exactly match what TitanicFeatureEngineer expects
    input_data = pd.DataFrame([{
        "Pclass":   pclass,
        "Name":     name,
        "Sex":      sex,
        "Age":      age,
        "SibSp":    sibsp,
        "Parch":    parch,
        "Ticket":   ticket,
        "Fare":     fare,
        "Cabin":    cabin,
        "Embarked": embarked,
    }])

    logger.info(
        f"Predicting for: Pclass={pclass}, Sex={sex}, Age={age}, "
        f"Fare={fare}, SibSp={sibsp}, Parch={parch}, Embarked={embarked}"
    )

    # ── Run through the full pipeline ────────────────────────
    # pipeline.predict() runs:
    # TitanicFeatureEngineer → ColumnTransformer → RandomForest
    # All in one call — no manual preprocessing here
    survived     = int(_pipeline.predict(input_data)[0])
    probability  = float(_pipeline.predict_proba(input_data)[0][1])
    interpretation = _interpret(probability)

    result = {
        "survived":       survived,
        "probability":    round(probability, 4),
        "interpretation": interpretation,
    }

    logger.info(f"Prediction result: {result}")
    return result


# ============================================================
# BATCH PREDICT — for Kaggle test set or bulk requests
# ============================================================
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a full DataFrame.
    DataFrame must contain the same columns as raw Kaggle test.csv

    Returns the input DataFrame with two new columns:
        Survived    : 0 or 1
        Probability : float
    """

    if _pipeline is None:
        raise RuntimeError(
            "Model is not loaded. Run src/train.py to generate model.pkl first."
        )

    df = df.copy()

    predictions  = _pipeline.predict(df)
    probabilities = _pipeline.predict_proba(df)[:, 1]

    df["Survived"]    = predictions
    df["Probability"] = probabilities.round(4)

    logger.info(f"Batch prediction complete: {len(df)} rows")
    return df


# ============================================================
# QUICK CLI TEST
# Run: python -m src.predict
# ============================================================
if __name__ == "__main__":
    print("\n--- Quick Prediction Test ---\n")

    test_cases = [
        # Should likely survive: 1st class female
        dict(pclass=1, sex="female", age=29, fare=100,
             sibsp=0, parch=0, embarked="C",
             name="Smith, Mrs. John"),

        # Should likely not survive: 3rd class male
        dict(pclass=3, sex="male", age=25, fare=7.5,
             sibsp=0, parch=0, embarked="S",
             name="Jones, Mr. William"),

        # Edge case: young boy 2nd class
        dict(pclass=2, sex="male", age=8, fare=25,
             sibsp=1, parch=2, embarked="S",
             name="Brown, Master. Tommy"),
    ]

    for i, case in enumerate(test_cases, 1):
        result = predict_survival(**case)
        print(f"Case {i}: {case['name']}")
        print(f"  Survived      : {result['survived']}")
        print(f"  Probability   : {result['probability']}")
        print(f"  Interpretation: {result['interpretation']}\n")

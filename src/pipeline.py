import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# CUSTOM FEATURE ENGINEERING TRANSFORMER
# This runs FIRST in the sklearn pipeline before any imputation
# ============================================================
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):

    TITLE_MAP = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Lady": "Rare",
        "Sir": "Rare", "Capt": "Rare", "Col": "Rare", "Major": "Rare",
        "Don": "Rare", "Jonkheer": "Rare", "Rev": "Rare", "Dr": "Rare"
    }

    SEX_MAP = {
        "male": "male", "m": "male", "man": "male",
        "female": "female", "f": "female", "woman": "female",
    }

    EMBARKED_MAP = {
        "c": "C", "cherbourg": "C",
        "q": "Q", "queenstown": "Q",
        "s": "S", "southampton": "S",
    }

    def fit(self, X, y=None):
        return self  # stateless transformer, nothing to learn

    def transform(self, X):
        df = X.copy()

        # ── Standardize text columns ──────────────────────────
        df["Sex"] = (
            df["Sex"].str.strip().str.lower()
            .map(self.SEX_MAP)
            .fillna("male")
        )
        df["Embarked"] = (
            df["Embarked"].str.strip().str.lower()
            .map(self.EMBARKED_MAP)
            .fillna("S")
        )
        df["Cabin"] = df["Cabin"].fillna("UNKNOWN").str.strip().str.upper()
        df["Name"]   = df["Name"].str.strip()
        df["Ticket"] = df["Ticket"].str.strip().str.upper()

        # ── Title ─────────────────────────────────────────────
        df["Title"] = (
            df["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
            .map(self.TITLE_MAP)
            .fillna("Rare")
        )

        # ── Deck ──────────────────────────────────────────────
        df["Deck"] = df["Cabin"].apply(self._extract_deck)

        # ── CabinKnown ────────────────────────────────────────
        df["CabinKnown"] = (df["Cabin"] != "UNKNOWN").astype(int)

        # ── Family features ───────────────────────────────────
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
        df["FamilyCategory"] = df["FamilySize"].apply(self._family_category)

        # ── Ticket prefix ─────────────────────────────────────
        df["TicketPrefix"] = df["Ticket"].apply(self._ticket_prefix)

        # ── Age-based features (may still be NaN here) ────────
        df["IsChild"]  = (df["Age"] < 16).astype(float)   # float keeps NaN
        df["IsMother"] = (
            (df["Sex"] == "female") &
            (df["Parch"] > 0) &
            (df["Age"] > 18) &
            (df["Title"] == "Mrs")
        ).astype(int)

        # ── Clip outliers (thresholds fixed, not data-driven) ─
        df["Fare"] = df["Fare"].clip(upper=300)
        df["Age"]  = df["Age"].clip(upper=80)

        # ── Fare per person (after clip, uses FamilySize from above) ──
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

        return df

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _extract_deck(cabin):
        if cabin == "UNKNOWN":
            return "U"
        first = str(cabin)[0]
        return first if first in "ABCDEFGT" else "U"

    @staticmethod
    def _family_category(size):
        if size == 1:
            return "Alone"
        elif size <= 4:
            return "Small"
        else:
            return "Large"

    @staticmethod
    def _ticket_prefix(ticket):
        parts = ticket.split()
        if len(parts) > 1:
            prefix = parts[0].replace("/", "").replace(".", "").upper()
            return prefix if len(prefix) <= 6 else "OTHER"
        return "NONE"


# ============================================================
# FEATURE LISTS
# These are the single source of truth for what goes into the model
# Import these in train.py, predict.py — never hardcode elsewhere
# ============================================================

NUMERIC_FEATURES = [
    "Age", "Fare", "FarePerPerson",
    "IsAlone", "CabinKnown", "IsChild", "IsMother",
]

CATEGORICAL_FEATURES = [
    "Pclass", "Sex", "Embarked",
    "Title", "Deck", "FamilyCategory", "TicketPrefix",
]

# Columns that must exist in raw input before feature engineering
RAW_INPUT_COLUMNS = [
    "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]

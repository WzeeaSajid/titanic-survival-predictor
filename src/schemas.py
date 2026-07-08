from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


# ============================================================
# INPUT SCHEMA
# Every field has:
#   - type annotation
#   - sensible default where applicable
#   - Field() with description and constraints
#   - validators for business logic Pydantic can't catch alone
# ============================================================
class PassengerInput(BaseModel):

    pclass: int = Field(
        ...,
        ge=1, le=3,
        description="Passenger class: 1 = First, 2 = Second, 3 = Third",
        examples=[1, 2, 3],
    )

    sex: str = Field(
        ...,
        description="Passenger sex: 'male' or 'female'",
        examples=["male", "female"],
    )

    age: float = Field(
        ...,
        gt=0, le=100,
        description="Age in years. Must be between 0 and 100.",
        examples=[29.0, 8.0, 45.5],
    )

    fare: float = Field(
        ...,
        ge=0, le=600,
        description="Ticket fare paid in British pounds. Must be >= 0.",
        examples=[7.25, 71.83, 512.33],
    )

    sibsp: int = Field(
        default=0,
        ge=0, le=8,
        description="Number of siblings or spouses aboard.",
        examples=[0, 1, 2],
    )

    parch: int = Field(
        default=0,
        ge=0, le=6,
        description="Number of parents or children aboard.",
        examples=[0, 1, 2],
    )

    embarked: str = Field(
        default="S",
        description="Port of embarkation: 'C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton",
        examples=["C", "Q", "S"],
    )

    # Optional fields — model handles missing gracefully
    cabin: str = Field(
        default="UNKNOWN",
        description="Cabin number (optional). Leave blank if unknown.",
        examples=["C85", "B28", "UNKNOWN"],
    )

    name: Optional[str] = Field(
        default=None,
        description=(
            "Passenger name in Titanic format: 'LastName, Title. FirstName'. "
            "Used to extract title (Mr, Mrs, Miss, Master, etc.). "
            "If not provided, title is inferred from sex and age."
        ),
        examples=["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina"],
    )

    ticket: str = Field(
        default="NONE",
        description="Ticket number (optional).",
        examples=["A/5 21171", "PC 17599", "NONE"],
    )

    # ── Field Validators ─────────────────────────────────────

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"male", "female", "m", "f", "man", "woman"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid sex '{v}'. Must be one of: male, female."
            )
        return normalized

    @field_validator("embarked")
    @classmethod
    def validate_embarked(cls, v: str) -> str:
        normalized = v.strip().upper()
        allowed = {"C", "Q", "S", "CHERBOURG", "QUEENSTOWN", "SOUTHAMPTON"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid embarked '{v}'. Must be C, Q, or S."
            )
        return normalized

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Age must be greater than 0.")
        if v > 100:
            raise ValueError("Age must be 100 or less.")
        return round(v, 1)

    @field_validator("fare")
    @classmethod
    def validate_fare(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Fare cannot be negative.")
        return round(v, 4)

    @field_validator("cabin", mode="before")
    @classmethod
    def validate_cabin(cls, v: Optional[str]) -> str:
        # Always return a string — pipeline expects string not None
        if v is None or v.strip() == "":
            return "UNKNOWN"
        return v.strip().upper()

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return None
        return v.strip()

    @field_validator("ticket", mode="before")
    @classmethod
    def validate_ticket(cls, v: Optional[str]) -> str:
        if v is None or v.strip() == "":
            return "NONE"
        return v.strip().upper()

    # ── Model-level Validator ─────────────────────────────────
    # Runs after all field validators pass
    # Use for cross-field logic

    @model_validator(mode="after")
    def infer_name_from_demographics(self) -> "PassengerInput":
        """
        NOTE: When name is synthesized from sex+age, Title becomes a
        deterministic function of Sex+Age, not independent signal.
        This inflates confidence for no-name API predictions.

        If name is not provided, construct a synthetic name
        so TitanicFeatureEngineer can still extract a Title.
        The caveat above applies.

        Logic:
          male   + age < 16  → Master
          male   + age >= 16 → Mr
          female + age < 16  → Miss
          female             → Mrs (default for adult female)
        """
        if self.name is None:
            sex = self.sex.lower() if self.sex else "male"
            age = self.age or 30

            if sex in ("male", "m", "man"):
                title = "Master" if age < 16 else "Mr"
                self.name = f"Unknown, {title}. Unknown"
            else:
                title = "Miss" if age < 16 else "Mrs"
                self.name = f"Unknown, {title}. Unknown"

        return self

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "pclass":   3,
                "sex":      "male",
                "age":      22.0,
                "fare":     7.25,
                "sibsp":    1,
                "parch":    0,
                "embarked": "S",
                "name":     "Braund, Mr. Owen Harris",
                "cabin":    None,
                "ticket":   "A/5 21171",
            }
        }
    }


# ============================================================
# OUTPUT SCHEMA
# ============================================================
class PredictionOutput(BaseModel):

    survived: int = Field(
        ...,
        description="Survival prediction: 0 = Did not survive, 1 = Survived",
        examples=[0, 1],
    )

    probability: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Probability of survival between 0.0 and 1.0",
        examples=[0.82, 0.14],
    )

    interpretation: str = Field(
        ...,
        description="Human readable interpretation of the prediction",
        examples=["Very likely to survive", "Very unlikely to survive"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "survived":       1,
                "probability":    0.82,
                "interpretation": "Very likely to survive",
            }
        }
    }


# ============================================================
# HEALTH CHECK SCHEMA
# ============================================================
class HealthResponse(BaseModel):

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status":       "ok",
                "model_loaded": True,
                "model_path":   "models/model.pkl",
            }
        }
    }

    status: str = Field(
        ...,
        description="Service status",
        examples=["ok"],
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        examples=[True],
    )

    model_path: str = Field(
        ...,
        description="Path to the loaded model file",
        examples=["models/model.pkl"],
    )


# ============================================================
# ERROR SCHEMA
# ============================================================
class ErrorResponse(BaseModel):

    detail: str = Field(
        ...,
        description="Error message explaining what went wrong",
        examples=["Age must be greater than 0."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Age must be greater than 0.",
            }
        }
    }

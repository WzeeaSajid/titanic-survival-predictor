# Titanic Survival Predictor

Predicts whether a Titanic passenger would have survived using a Random Forest model trained on the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).

## Live Demo

- **Frontend** — http://13.60.22.65:8000
- **API docs** — http://13.60.22.65:8000/docs

## Model Performance

| Metric | Score |
|--------|-------|
| CV Accuracy | 84.28% |
| Precision | 0.81 |
| Recall | 0.74 |
| F1 Score | 0.77 |
| ROC-AUC | 0.88 |

## Project Structure

```
├── api/
│   └── main.py              # FastAPI app with /predict endpoint
├── src/
│   ├── pipeline.py           # TitanicFeatureEngineer transformer
│   ├── train.py              # Training pipeline (RandomizedSearchCV)
│   ├── predict.py            # Module-level model loading + prediction
│   └── schemas.py            # Pydantic v2 request/response schemas
├── tests/
│   └── test_api.py           # 38 API endpoint tests
├── static/
│   └── index.html            # Custom HTML/CSS frontend
├── models/
│   └── model.pkl             # Trained sklearn Pipeline
├── data/
│   └── Titanic-Dataset.csv   # Training data (891 rows)
├── Dockerfile                # Multi-stage Docker build
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train

# Run the API
uvicorn api.main:app --reload --port 8000

# Run tests
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t titanic-predictor .
docker run -p 8000:8000 titanic-predictor
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "fare": 100,
    "sibsp": 0,
    "parch": 0,
    "embarked": "C"
  }'
```

Response:
```json
{
  "survived": 1,
  "probability": 0.95,
  "interpretation": "Very likely to survive"
}
```

## Tech Stack

- **Backend** — FastAPI, scikit-learn, Pydantic v2
- **Frontend** — Custom HTML/CSS with period-themed design
- **Model** — RandomForest with `class_weight="balanced"`, tuned via RandomizedSearchCV
- **Deployment** — Docker multi-stage build, non-root user, healthcheck

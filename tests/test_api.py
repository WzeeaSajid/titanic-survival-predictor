import pytest
from fastapi.testclient import TestClient
from api.main import app

# ============================================================
# TEST CLIENT
# FastAPI's TestClient runs the app in a real ASGI environment
# No server needed — tests run purely in memory
# ============================================================
client = TestClient(app)


# ============================================================
# FIXTURES
# Reusable test data
# ============================================================
@pytest.fixture
def valid_first_class_female():
    """1st class female — should have high survival probability"""
    return {
        "pclass":   1,
        "sex":      "female",
        "age":      29.0,
        "fare":     100.0,
        "sibsp":    0,
        "parch":    0,
        "embarked": "C",
        "name":     "Smith, Mrs. John",
    }


@pytest.fixture
def valid_third_class_male():
    """3rd class male — should have low survival probability"""
    return {
        "pclass":   3,
        "sex":      "male",
        "age":      25.0,
        "fare":     7.25,
        "sibsp":    0,
        "parch":    0,
        "embarked": "S",
    }


@pytest.fixture
def valid_child():
    """Young boy 2nd class with family"""
    return {
        "pclass":   2,
        "sex":      "male",
        "age":      8.0,
        "fare":     25.0,
        "sibsp":    1,
        "parch":    2,
        "embarked": "S",
        "name":     "Brown, Master. Tommy",
    }


@pytest.fixture
def minimal_input():
    """Only required fields — tests defaults work correctly"""
    return {
        "pclass":   2,
        "sex":      "female",
        "age":      30.0,
        "fare":     20.0,
        "embarked": "Q",
    }


# ============================================================
# SYSTEM TESTS
# ============================================================
class TestSystemEndpoints:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_frontend_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]
        assert "Survival manifest" in response.text
        assert "class=\"ticket\"" in response.text

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "model_path"   in data

    def test_health_status_is_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_model_is_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True, (
            "Model is not loaded. Run src/train.py first."
        )

    def test_docs_endpoint_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200


# ============================================================
# PREDICTION TESTS — VALID INPUTS
# ============================================================
class TestPredictValidInputs:

    def test_predict_returns_200(self, valid_first_class_female):
        response = client.post("/predict", json=valid_first_class_female)
        assert response.status_code == 200

    def test_predict_response_structure(self, valid_first_class_female):
        response = client.post("/predict", json=valid_first_class_female)
        data = response.json()
        assert "survived"       in data
        assert "probability"    in data
        assert "interpretation" in data

    def test_predict_survived_is_binary(self, valid_first_class_female):
        response = client.post("/predict", json=valid_first_class_female)
        data = response.json()
        assert data["survived"] in [0, 1]

    def test_predict_probability_is_valid_float(self, valid_first_class_female):
        response = client.post("/predict", json=valid_first_class_female)
        data = response.json()
        prob = data["probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_interpretation_is_string(self, valid_first_class_female):
        response = client.post("/predict", json=valid_first_class_female)
        data = response.json()
        assert isinstance(data["interpretation"], str)
        assert len(data["interpretation"]) > 0

    def test_first_class_female_survives(self, valid_first_class_female):
        """Domain knowledge: 1st class females had very high survival rate"""
        response = client.post("/predict", json=valid_first_class_female)
        data = response.json()
        assert data["survived"] == 1, (
            "1st class female should survive — check model training"
        )

    def test_third_class_male_does_not_survive(self, valid_third_class_male):
        """Domain knowledge: 3rd class males had very low survival rate"""
        response = client.post("/predict", json=valid_third_class_male)
        data = response.json()
        assert data["survived"] == 0, (
            "3rd class male should not survive — check model training"
        )

    def test_minimal_input_works(self, minimal_input):
        """Only required fields — all optional fields use defaults"""
        response = client.post("/predict", json=minimal_input)
        assert response.status_code == 200

    def test_child_prediction_works(self, valid_child):
        response = client.post("/predict", json=valid_child)
        assert response.status_code == 200
        data = response.json()
        assert data["survived"] in [0, 1]

    def test_predict_with_cabin(self, valid_first_class_female):
        payload = {**valid_first_class_female, "cabin": "C85"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_with_ticket(self, valid_first_class_female):
        payload = {**valid_first_class_female, "ticket": "PC 17599"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_all_embarkation_ports(self):
        base = {
            "pclass": 2, "sex": "female",
            "age": 30.0, "fare": 20.0,
        }
        for port in ["C", "Q", "S"]:
            response = client.post("/predict", json={**base, "embarked": port})
            assert response.status_code == 200, f"Failed for embarked={port}"

    def test_predict_all_passenger_classes(self):
        base = {
            "sex": "male", "age": 30.0,
            "fare": 20.0, "embarked": "S",
        }
        for pclass in [1, 2, 3]:
            response = client.post("/predict", json={**base, "pclass": pclass})
            assert response.status_code == 200, f"Failed for pclass={pclass}"


# ============================================================
# PREDICTION TESTS — INVALID INPUTS
# Every invalid case must return 422 Unprocessable Entity
# ============================================================
class TestPredictInvalidInputs:

    def test_invalid_pclass_too_low(self):
        payload = {
            "pclass": 0, "sex": "male",
            "age": 25.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_pclass_too_high(self):
        payload = {
            "pclass": 4, "sex": "male",
            "age": 25.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_sex(self):
        payload = {
            "pclass": 1, "sex": "robot",
            "age": 25.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_zero(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 0.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_negative(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": -5.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_too_high(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 150.0, "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_fare_negative(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 25.0, "fare": -10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_embarked(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 25.0, "fare": 10.0, "embarked": "X",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_field_pclass(self):
        payload = {
            "sex": "male", "age": 25.0,
            "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_field_age(self):
        payload = {
            "pclass": 1, "sex": "male",
            "fare": 10.0, "embarked": "S",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_extra_fields_ignored(self):
        """Extra fields should be silently ignored, not cause errors"""
        payload = {
            "pclass": 1, "sex": "female",
            "age": 29.0, "fare": 100.0, "embarked": "C",
            "unknown_field": "should_be_ignored",
            "another_field": 999,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_sibsp_too_high(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 25.0, "fare": 10.0,
            "embarked": "S", "sibsp": 99,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_parch_too_high(self):
        payload = {
            "pclass": 1, "sex": "male",
            "age": 25.0, "fare": 10.0,
            "embarked": "S", "parch": 99,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ============================================================
# MODEL INFO TESTS
# ============================================================
class TestModelInfo:

    def test_model_info_returns_200(self):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_contains_metrics(self):
        response = client.get("/model-info")
        data = response.json()
        assert "metrics"     in data
        assert "trained_at"  in data
        assert "best_params" in data

    def test_model_info_metrics_have_accuracy(self):
        response = client.get("/model-info")
        data = response.json()
        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.70, (
            f"Model accuracy {metrics['accuracy']} is too low. "
            "Retrain with better hyperparameters."
        )

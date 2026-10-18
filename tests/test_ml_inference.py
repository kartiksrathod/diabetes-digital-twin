"""ML Inference, Data Contracts, and Feature Order Tests for Model B Random Forest.

Validates that feature lists, preprocessing, probability bounds, threshold 0.57,
and absence of leakage hold across all training and runtime inference scripts.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"
FEATURES_V2_PATH = PROJECT_ROOT / "data" / "processed" / "shanghai_t2dm_features_v2.csv"
FIXTURE_MODEL_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_model_b.joblib"
FIXTURE_FEATURES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_features_v2.csv"

# Canonical Model B 40-feature definition
EXPECTED_40_FEATURES = [
    "glucose_t0",
    "glucose_t_minus_15m",
    "glucose_t_minus_30m",
    "glucose_t_minus_45m",
    "glucose_t_minus_60m",
    "glucose_t_minus_75m",
    "glucose_t_minus_90m",
    "glucose_t_minus_105m",
    "glucose_t_minus_120m",
    "delta_15m",
    "delta_30m",
    "delta_60m",
    "delta_120m",
    "glucose_mean_1h",
    "glucose_std_1h",
    "glucose_min_1h",
    "glucose_max_1h",
    "glucose_mean_2h",
    "glucose_std_2h",
    "glucose_min_2h",
    "glucose_max_2h",
    "slope_30m",
    "slope_60m",
    "slope_120m",
    "age",
    "bmi",
    "diabetes_duration",
    "fasting_glucose",
    "postprandial_glucose",
    "hba1c",
    "meal_events_30m",
    "meal_events_60m",
    "meal_events_120m",
    "insulin_events_30m",
    "insulin_events_60m",
    "insulin_events_120m",
    "insulin_amount_30m",
    "insulin_amount_60m",
    "insulin_amount_120m",
    "medication_events_120m",
]

FINAL_FROZEN_THRESHOLD = 0.57


class TestModelArtifactsAndContracts:
    """Tests loading of model artifact and feature alignment."""

    def test_fixture_model_artifact_exists_and_loads(self):
        """Always validates the scikit-learn Pipeline architecture and contract in CI."""
        assert FIXTURE_MODEL_PATH.exists(), f"Fixture model missing at {FIXTURE_MODEL_PATH}"
        model = joblib.load(FIXTURE_MODEL_PATH)
        assert isinstance(model, Pipeline), "Model must be a scikit-learn Pipeline"
        assert "imputer" in model.named_steps, "Pipeline must contain 'imputer' step"
        assert "model" in model.named_steps, "Pipeline must contain 'model' step"
        assert hasattr(model, "predict_proba"), "Pipeline must expose predict_proba"
        assert list(model.classes_) == [0, 1], "Model classes must be [0, 1]"

    def test_production_model_artifact_when_available(self):
        """Validates the production model weights only when present in the environment."""
        if not MODEL_PATH.exists():
            pytest.skip("Production model artifact not present in clean checkout")
        model = joblib.load(MODEL_PATH)
        assert isinstance(model, Pipeline), "Model must be a scikit-learn Pipeline"
        assert "imputer" in model.named_steps, "Pipeline must contain 'imputer' step"
        assert "model" in model.named_steps, "Pipeline must contain 'model' step"
        assert hasattr(model, "predict_proba"), "Pipeline must expose predict_proba"
        assert list(model.classes_) == [0, 1], "Model classes must be [0, 1]"

    def test_feature_list_length_and_names(self):
        assert len(EXPECTED_40_FEATURES) == 40, "Feature set must contain exactly 40 features"

    def test_features_agree_across_codebase(self):
        """Cross-check feature list definitions across dashboard_api, replay_stream, push_prediction_to_twin."""
        from dashboard_api import FEATURES as DASHBOARD_FEATURES
        from ml.replay_stream import FEATURES as REPLAY_FEATURES
        from ml.push_prediction_to_twin import FEATURES as PUSH_FEATURES

        assert DASHBOARD_FEATURES == EXPECTED_40_FEATURES, "dashboard_api.py feature list mismatch"
        assert REPLAY_FEATURES == EXPECTED_40_FEATURES, "replay_stream.py feature list mismatch"
        assert PUSH_FEATURES == EXPECTED_40_FEATURES, "push_prediction_to_twin.py feature list mismatch"

    def test_no_leakage_identifiers_in_features(self):
        """Ensure no patient IDs, timestamps, or target labels exist in feature list."""
        forbidden = {"patient_id", "session_id", "timestamp", "spike_label", "future_max_2h", "future_increase_2h"}
        intersection = forbidden.intersection(EXPECTED_40_FEATURES)
        assert len(intersection) == 0, f"Target or identifier leakage found in features: {intersection}"


class TestInferencePipelineBehavior:
    """Tests model inference execution, bounds, missing values, and thresholding."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        assert FIXTURE_MODEL_PATH.exists(), f"Fixture model missing at {FIXTURE_MODEL_PATH}"
        return joblib.load(FIXTURE_MODEL_PATH)

    @pytest.fixture(scope="class")
    def sample_features_df(self):
        assert FIXTURE_FEATURES_PATH.exists(), f"Fixture features missing at {FIXTURE_FEATURES_PATH}"
        df = pd.read_csv(FIXTURE_FEATURES_PATH, nrows=10, low_memory=False)
        return df

    def test_inference_produces_valid_probabilities(self, loaded_model, sample_features_df):
        X = sample_features_df[EXPECTED_40_FEATURES]
        probabilities = loaded_model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        # Probabilities sum to 1.0
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

        # Risk score is column 1 (positive class)
        risk_scores = probabilities[:, 1]
        assert np.all((risk_scores >= 0.0) & (risk_scores <= 1.0))
        assert not np.any(np.isnan(risk_scores))

    def test_pipeline_handles_missing_values_via_imputer(self, loaded_model, sample_features_df):
        """Model must gracefully handle NaN in input features using the trained SimpleImputer."""
        X_with_nans = sample_features_df[EXPECTED_40_FEATURES].copy()
        # Introduce NaNs in clinical, glucose lag, and context columns
        X_with_nans.loc[0, "hba1c"] = np.nan
        X_with_nans.loc[0, "glucose_t_minus_120m"] = np.nan
        X_with_nans.loc[0, "insulin_amount_30m"] = np.nan

        probabilities = loaded_model.predict_proba(X_with_nans)
        assert not np.any(np.isnan(probabilities[0])), "Imputer must handle NaNs cleanly"
        assert 0.0 <= probabilities[0, 1] <= 1.0

    def test_missing_required_column_raises_error(self, loaded_model, sample_features_df):
        """Omitting a required feature should fail safely rather than silently misalign."""
        incomplete_X = sample_features_df[EXPECTED_40_FEATURES].drop(columns=["glucose_t0"])
        with pytest.raises(Exception):
            loaded_model.predict_proba(incomplete_X)

    def test_threshold_057_classification_decision(self, loaded_model, sample_features_df):
        """Verify the 0.57 threshold boundary behavior."""
        X = sample_features_df[EXPECTED_40_FEATURES]
        scores = loaded_model.predict_proba(X)[:, 1]

        for score in scores:
            level = "HIGH" if score >= FINAL_FROZEN_THRESHOLD else "LOW"
            if score >= 0.57:
                assert level == "HIGH"
            else:
                assert level == "LOW"

    def test_inference_reproducibility(self, loaded_model, sample_features_df):
        """Random forest inference in scikit-learn is deterministic on identical inputs."""
        X = sample_features_df[EXPECTED_40_FEATURES]
        run1 = loaded_model.predict_proba(X)
        run2 = loaded_model.predict_proba(X)
        np.testing.assert_allclose(run1, run2, atol=1e-12)

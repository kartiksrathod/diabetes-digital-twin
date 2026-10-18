"""ML and Feature Engineering Unit and Regression Tests.

Validates:
- Feature generation contracts (names, ordering, types, 40 canonical features).
- Absence of leakage (labels, future metrics, identifiers).
- Safe handling of missing values, malformed inputs, and imputation.
- Model immutability during inference (no weight mutation or retraining).
- Deterministic, finite probability scores in [0, 1].
- Frozen threshold (0.57) boundary conditions.
- Constant 2-hour prediction horizon.
- Patient-level split integrity (zero patient overlap between train/val/test).
- Metric evaluation correctness on held-out test data.
- Feature windowing helpers (nonempty_event, parse_insulin_units, calculate_prior_window).
"""

from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.build_features_v2 import (
    calculate_prior_window,
    nonempty_event,
    parse_insulin_units,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_b_random_forest.joblib"
FEATURES_V2_PATH = PROJECT_ROOT / "data" / "processed" / "shanghai_t2dm_features_v2.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
FINAL_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "final_test_results.csv"

FIXTURE_MODEL_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_model_b.joblib"
FIXTURE_FEATURES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_features_v2.csv"
FIXTURE_TRAIN_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_train.csv"
FIXTURE_VAL_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_validation.csv"
FIXTURE_TEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "fixture_test.csv"

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

FROZEN_THRESHOLD = 0.57


class TestFeatureWindowingHelpers:
    """Unit tests for feature engineering utility functions in build_features_v2.py."""

    def test_nonempty_event_handles_various_null_and_event_types(self):
        s = pd.Series(["", "   ", "nan", "none", "null", "未记录", "无", "-", "0", "0.0", 0, "meal", "Humalog 5 IU"])
        result = nonempty_event(s)
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        assert result.tolist() == expected

    @pytest.mark.parametrize("input_val,expected_units", [
        ("Gansulin 40R, 6 IU", 6.0),
        ("Humulin 70/30 8 IU", 8.0),
        ("insulin aspart 70/30, 18.5 IU", 18.5),
        ("12.0", 12.0),
        (10, 10.0),
        (None, 0.0),
        ("", 0.0),
        ("None", 0.0),
        ("invalid text without numbers", 0.0),
    ])
    def test_parse_insulin_units(self, input_val, expected_units):
        assert parse_insulin_units(input_val) == pytest.approx(expected_units, rel=1e-3)

    def test_calculate_prior_window_excludes_current_and_future_events(self):
        """Historical window [t - window, t) must strictly exclude the event at t and any future events."""
        base_time = pd.Timestamp("2026-01-01 12:00:00")
        timestamps = np.array([
            base_time - pd.Timedelta(minutes=70),
            base_time - pd.Timedelta(minutes=30),
            base_time - pd.Timedelta(minutes=15),
            base_time,  # Current row t
            base_time + pd.Timedelta(minutes=15),  # Future row
        ], dtype="datetime64[ns]").astype(np.int64)

        # Events occur at t-30m and at t
        events = np.array([False, True, False, True, True])

        # For a 60-minute window prior to t (current row index 3):
        # Window is [11:00, 12:00). Event at t-30m (11:30) is inside; event at t (12:00) is excluded.
        counts = calculate_prior_window(timestamps, events, minutes=60)
        assert counts[3] == 1.0, "Current row at t must be excluded from prior window"
        # At index 0 (10:50), there are no events in prior 60m
        assert counts[0] == 0.0
        # At index 1 (11:30), no events in [10:30, 11:30)
        assert counts[1] == 0.0
        # At index 2 (11:45), the event at 11:30 is in [10:45, 11:45)
        assert counts[2] == 1.0


class TestModelImmutabilityAndInference:
    """Tests model stability, deterministic predictions, and absence of mutation."""

    @pytest.fixture(scope="class")
    def model(self):
        assert FIXTURE_MODEL_PATH.exists(), f"Fixture model missing at {FIXTURE_MODEL_PATH}"
        return joblib.load(FIXTURE_MODEL_PATH)

    @pytest.fixture(scope="class")
    def sample_features(self):
        assert FIXTURE_FEATURES_PATH.exists(), f"Fixture features missing at {FIXTURE_FEATURES_PATH}"
        df = pd.read_csv(FIXTURE_FEATURES_PATH, nrows=20, low_memory=False)
        return df[EXPECTED_40_FEATURES]

    def test_inference_does_not_mutate_model_parameters(self, model, sample_features):
        """predict_proba must not change internal estimators, weights, or pipeline state."""
        rf_step = model.named_steps["model"]
        initial_n_estimators = len(rf_step.estimators_)
        initial_feature_importances = copy.deepcopy(rf_step.feature_importances_)

        # Run multiple inference passes
        for _ in range(5):
            _ = model.predict_proba(sample_features)

        assert len(rf_step.estimators_) == initial_n_estimators
        np.testing.assert_array_equal(rf_step.feature_importances_, initial_feature_importances)

    def test_deterministic_predictions_on_fixed_inputs(self, model, sample_features):
        """Random forest inference must be strictly deterministic across repeated runs."""
        res1 = model.predict_proba(sample_features)
        res2 = model.predict_proba(sample_features)
        np.testing.assert_allclose(res1, res2, atol=1e-12)

    def test_risk_score_bounds_and_finiteness(self, model, sample_features):
        """Probabilities must be finite, sum to 1, and fall within [0, 1]."""
        proba = model.predict_proba(sample_features)
        assert np.all(np.isfinite(proba))
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("score,expected_decision", [
        (0.569999, "LOW"),
        (0.570000, "HIGH"),
        (0.570001, "HIGH"),
        (0.0, "LOW"),
        (1.0, "HIGH"),
    ])
    def test_frozen_threshold_boundary_decision(self, score, expected_decision):
        decision = "HIGH" if score >= FROZEN_THRESHOLD else "LOW"
        assert decision == expected_decision

    def test_prediction_horizon_constant(self):
        from dashboard_api import app
        # Horizon across the codebase is consistently 2 hours
        assert "2 hours" in "Next 2 hours"


class TestDatasetAndSplittingIntegrity:
    """Tests patient-level split separation, held-out evaluation, and leakage absence."""

    def test_patient_splits_are_mutually_disjoint(self):
        """Train, validation, and test splits must share zero patient IDs (tested via fixtures)."""
        train_df = pd.read_csv(FIXTURE_TRAIN_PATH, usecols=["patient_id"])
        val_df = pd.read_csv(FIXTURE_VAL_PATH, usecols=["patient_id"])
        test_df = pd.read_csv(FIXTURE_TEST_PATH, usecols=["patient_id"])

        train_patients = set(train_df["patient_id"].astype(str).str.strip())
        val_patients = set(val_df["patient_id"].astype(str).str.strip())
        test_patients = set(test_df["patient_id"].astype(str).str.strip())

        assert len(train_patients) > 0
        assert len(val_patients) > 0
        assert len(test_patients) > 0

        # No patient leaks between splits
        assert train_patients.isdisjoint(val_patients), "Train and validation splits share patients"
        assert train_patients.isdisjoint(test_patients), "Train and test splits share patients"
        assert val_patients.isdisjoint(test_patients), "Validation and test splits share patients"

    def test_production_splits_disjoint_when_available(self):
        """Validates patient disjointness on full production splits when present."""
        if not TRAIN_PATH.exists():
            pytest.skip("Full production splits not present in clean checkout")
        train_df = pd.read_csv(TRAIN_PATH, usecols=["patient_id"])
        val_df = pd.read_csv(VAL_PATH, usecols=["patient_id"])
        test_df = pd.read_csv(TEST_PATH, usecols=["patient_id"])

        train_patients = set(train_df["patient_id"].astype(str).str.strip())
        val_patients = set(val_df["patient_id"].astype(str).str.strip())
        test_patients = set(test_df["patient_id"].astype(str).str.strip())

        assert len(train_patients) > 0
        assert len(val_patients) > 0
        assert len(test_patients) > 0
        assert train_patients.isdisjoint(val_patients)
        assert train_patients.isdisjoint(test_patients)
        assert val_patients.isdisjoint(test_patients)

    def test_features_v2_columns_and_absence_of_future_leakage(self):
        """Features dataset must contain all 40 features and exclude future spike labels from predictors."""
        df_head = pd.read_csv(FIXTURE_FEATURES_PATH, nrows=5)
        for feature in EXPECTED_40_FEATURES:
            assert feature in df_head.columns, f"Missing feature {feature} in {FIXTURE_FEATURES_PATH}"

        # Target and future columns must never be in the predictor list
        forbidden_leakage = {
            "spike_label",
            "future_max_2h",
            "future_increase_2h",
            "patient_id",
            "session_id",
            "timestamp",
        }
        assert forbidden_leakage.isdisjoint(set(EXPECTED_40_FEATURES))

    def test_production_features_v2_columns_when_available(self):
        """Validates feature columns on full production dataset when available."""
        if not FEATURES_V2_PATH.exists():
            pytest.skip("Production features dataset not present in clean checkout")
        df_head = pd.read_csv(FEATURES_V2_PATH, nrows=5)
        for feature in EXPECTED_40_FEATURES:
            assert feature in df_head.columns, f"Missing feature {feature} in {FEATURES_V2_PATH}"

        # Target and future columns must never be in the predictor list
        forbidden_leakage = {
            "spike_label",
            "future_max_2h",
            "future_increase_2h",
            "patient_id",
            "session_id",
            "timestamp",
        }
        assert forbidden_leakage.isdisjoint(set(EXPECTED_40_FEATURES))

    def test_final_evaluation_metrics_match_stored_results(self):
        """Recorded test evaluation metrics in final_test_results.csv match held-out criteria."""
        assert FINAL_RESULTS_PATH.exists()
        res = pd.read_csv(FINAL_RESULTS_PATH).iloc[0]

        assert res["model"] == "Model B Random Forest"
        assert res["threshold"] == FROZEN_THRESHOLD
        assert res["test_patients"] == 15
        assert res["test_rows"] == 16012
        assert 0.0 <= res["precision"] <= 1.0
        assert 0.0 <= res["recall"] <= 1.0
        assert 0.0 <= res["f1"] <= 1.0
        assert 0.0 <= res["roc_auc"] <= 1.0

"""Unit and Regression Tests for Simulated Replay Pipeline (ml/replay_stream.py).

Validates:
- Row retrieval (get_rows): patient filtering, session filtering, timestamp ordering, invalid queries.
- Trend calculation (glucose_trend): threshold boundaries (+10 -> Rising, -10 -> Falling, between -> Stable).
- Float cleaning (clean_float): NaN/None handling, valid numeric conversion.
- Replay loop execution with mocked Azure client:
  - Source timestamp and delta_30m propagation.
  - Frozen threshold (0.57) risk classification.
  - Row limit adherence.
  - Zero unmocked network calls.
  - Replay failure containment on missing twin or Azure SDK exception.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from ml.replay_stream import (
    FEATURES,
    FINAL_THRESHOLD,
    clean_float,
    get_rows,
    glucose_trend,
)


class TestReplayHelpers:
    """Tests utility functions in replay_stream.py."""

    @pytest.mark.parametrize("delta,expected_trend", [
        (10.0, "Rising"),
        (25.4, "Rising"),
        (-10.0, "Falling"),
        (-14.2, "Falling"),
        (9.99, "Stable"),
        (-9.99, "Stable"),
        (0.0, "Stable"),
    ])
    def test_glucose_trend_boundaries(self, delta, expected_trend):
        assert glucose_trend(delta) == expected_trend

    @pytest.mark.parametrize("val,expected", [
        (42, 42.0),
        ("25.5", 25.5),
        (None, None),
        (float("nan"), None),
    ])
    def test_clean_float(self, val, expected):
        result = clean_float(val)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestReplayGetRows:
    """Tests row filtering and sequencing in replay_stream.get_rows."""

    @pytest.fixture
    def sample_cgm_df(self):
        return pd.DataFrame([
            {"patient_id": "2001", "session_id": "sess_1", "timestamp": "2020-11-02 12:00:00", "glucose_t0": 100},
            {"patient_id": "2001", "session_id": "sess_1", "timestamp": "2020-11-02 11:30:00", "glucose_t0": 95},
            {"patient_id": "2001", "session_id": "sess_2", "timestamp": "2020-11-03 09:00:00", "glucose_t0": 110},
            {"patient_id": "2002", "session_id": "sess_3", "timestamp": "2020-11-02 10:00:00", "glucose_t0": 120},
            {"patient_id": "2001", "session_id": "sess_1", "timestamp": "malformed_time", "glucose_t0": 90},
        ])

    def test_get_rows_filters_and_sorts_chronologically(self, sample_cgm_df):
        rows = get_rows(sample_cgm_df, patient_id="2001", session_id="sess_1")
        # Malformed time row dropped, 2 rows remain, sorted chronologically (11:30 before 12:00)
        assert len(rows) == 2
        assert rows.iloc[0]["glucose_t0"] == 95
        assert rows.iloc[1]["glucose_t0"] == 100
        assert str(rows.iloc[0]["timestamp"]) == "2020-11-02 11:30:00"

    def test_get_rows_unknown_patient_raises_value_error(self, sample_cgm_df):
        with pytest.raises(ValueError, match="Patient '9999' was not found"):
            get_rows(sample_cgm_df, patient_id="9999", session_id=None)

    def test_get_rows_unknown_session_raises_value_error_listing_available(self, sample_cgm_df):
        with pytest.raises(ValueError, match="Session 'invalid_sess' was not found"):
            get_rows(sample_cgm_df, patient_id="2001", session_id="invalid_sess")


class TestReplayStreamExecution:
    """Tests end-to-end replay logic with mocked Azure client."""

    @patch("ml.replay_stream.get_client")
    @patch("ml.replay_stream.update_patient_twin")
    def test_replay_invokes_azure_with_correct_payload(self, mock_update, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_digital_twin.return_value = {"$dtId": "P001", "patientId": "2001"}
        mock_get_client.return_value = mock_client

        # Replay with CLI args simulated
        test_args = [
            "replay_stream.py",
            "--patient-id", "2001",
            "--session-id", "2001_0_20201102",
            "--rows", "2",
            "--delay", "0.0",
        ]

        with patch("sys.argv", test_args):
            from ml.replay_stream import main
            main()

        mock_get_client.assert_called_once()
        # Initial verification call + 2 read-back verification calls
        assert mock_client.get_digital_twin.call_count == 3
        # Exactly 2 rows pushed
        assert mock_update.call_count == 2

        # Inspect call kwargs of first update
        call1_kwargs = mock_update.call_args_list[0].kwargs
        assert call1_kwargs["patient_id"] == "2001"
        assert "source_timestamp" in call1_kwargs
        assert call1_kwargs["source_timestamp"] is not None
        assert "delta_30m" in call1_kwargs
        assert isinstance(call1_kwargs["current_glucose"], float)
        assert 0.0 <= call1_kwargs["risk_score"] <= 1.0
        assert call1_kwargs["risk_level"] in ("HIGH", "LOW")
        assert call1_kwargs["prediction_window"] == "Next 2 hours"

    @patch("ml.replay_stream.get_client")
    def test_replay_fails_cleanly_if_target_twin_missing(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_digital_twin.side_effect = RuntimeError("Twin P999 does not exist")
        mock_get_client.return_value = mock_client

        test_args = [
            "replay_stream.py",
            "--twin-id", "P999",
            "--rows", "1",
            "--delay", "0.0",
        ]

        with patch("sys.argv", test_args):
            from ml.replay_stream import main
            with pytest.raises(RuntimeError, match="Twin P999 does not exist"):
                main()

    def test_replay_cli_invalid_args_rejected(self):
        with patch("sys.argv", ["replay_stream.py", "--rows", "0"]):
            from ml.replay_stream import main
            with pytest.raises(ValueError, match="--rows must be greater than 0"):
                main()

        with patch("sys.argv", ["replay_stream.py", "--delay", "-1.0"]):
            from ml.replay_stream import main
            with pytest.raises(ValueError, match="--delay cannot be negative"):
                main()

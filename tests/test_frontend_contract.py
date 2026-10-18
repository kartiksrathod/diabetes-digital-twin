"""Automated Contract & Static Verification Tests for dashboard/index.html.

Inspects the frontend implementation for API contract alignment, DOM element presence,
clinical safety guardrails, frozen decision threshold, timeout handling, and XSS safety.
"""

from pathlib import Path
import re
import pytest

INDEX_HTML_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


@pytest.fixture(scope="module")
def index_html_content():
    assert INDEX_HTML_PATH.exists(), f"dashboard/index.html not found at {INDEX_HTML_PATH}"
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


class TestFrontendContracts:
    """Verifies JavaScript and HTML contracts in dashboard/index.html."""

    def test_api_base_and_endpoints(self, index_html_content):
        assert 'const API_BASE = "http://127.0.0.1:8000/api";' in index_html_content
        assert "/twin-state/P001" in index_html_content
        assert "/replay-history" in index_html_content
        assert "/system-status" in index_html_content

    def test_mode_live_query_parameter_handling(self, index_html_content):
        """Verifies that ?mode=live genuinely toggles live mode."""
        assert 'new URLSearchParams(window.location.search).get("mode") === "live"' in index_html_content

    def test_frozen_threshold_in_frontend(self, index_html_content):
        """Verifies threshold is frozen at 0.57 in JavaScript."""
        match = re.search(r"const THRESHOLD\s*=\s*(0\.570*);", index_html_content)
        assert match is not None, "THRESHOLD = 0.5700 not found in frontend JS"

    def test_clinical_safety_guardrails_present(self, index_html_content):
        """Frontend must display clinical safety guardrail notice."""
        assert "Safety Guardrail" in index_html_content
        assert "strictly outputs observational context and model risk probabilities" in index_html_content
        assert "does not provide medical diagnosis, insulin dosing" in index_html_content

    def test_data_provenance_explicitly_disclosed(self, index_html_content):
        """Frontend must not claim live wearable sensor data."""
        assert "Simulated replay of historical CGM data" in index_html_content
        assert "live wearable telemetry" not in index_html_content.lower()

    def test_essential_dom_elements_exist(self, index_html_content):
        """Verify all DOM element IDs manipulated by JS exist in the HTML."""
        required_ids = [
            "step-slider", "step-counter", "replay-time-label", "header-timestamp",
            "btn-play", "btn-prev", "btn-next", "btn-reset", "btn-refresh",
            "mode-badge", "header-session", "card-predicted-event",
            "card-glucose", "card-delta", "card-trend", "card-risk-score", "card-margin",
            "card-risk-container", "badge-classification", "card-class-text",
            "decision-math-badge", "decision-bar-fill", "math-score", "math-delta", "math-class",
            "glucose-svg", "risk-svg", "status-azure", "status-mcp", "status-agent", "status-replay"
        ]
        for element_id in required_ids:
            assert f'id="{element_id}"' in index_html_content or f"id='{element_id}'" in index_html_content, \
                f"Missing DOM element id='{element_id}' in index.html"

    def test_xss_safe_rendering_of_dynamic_values(self, index_html_content):
        """Ensure numerical card values use innerText / textContent rather than innerHTML."""
        assert "cardGlucose.innerText" in index_html_content
        assert "cardRiskScore.innerText" in index_html_content
        assert "cardMargin.innerText" in index_html_content
        assert "cardTrend.innerText" in index_html_content

    def test_fetch_has_abort_controller_timeout(self, index_html_content):
        """Regression test for API fetch timeout protection using AbortController."""
        assert "new AbortController()" in index_html_content
        assert "signal: controller.signal" in index_html_content
        assert "Request timed out after" in index_html_content
        assert "clearTimeout(timeoutId)" in index_html_content

    def test_dashboard_precision_metric_reflects_model_b(self, index_html_content):
        """Regression test ensuring dashboard displays Model B test precision (43.45%), not outdated Model A (29.25%)."""
        assert "43.45%" in index_html_content
        assert "Precision is 29.25%" not in index_html_content

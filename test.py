"""
Unit tests for the customer segmentation dashboard's core logic.

These target the pure/testable functions in app/dashboard.py — data
normalization, segment lookup helpers, revenue aggregation, and the
runtime cluster-to-label mapping. UI-rendering functions (st.* calls)
are intentionally not covered here since they require a live Streamlit
runtime; the goal is to lock down the business logic that would silently
produce wrong numbers or mislabeled segments if it broke.

Run with:
    pytest test_dashboard.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the app package importable when tests are run from the repo root
# or from inside app/.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "app"))

import app.dashboard as dash  # noqa: E402


# --------------------------------------------------------------------------
# normalize_columns
# --------------------------------------------------------------------------

class TestNormalizeColumns:
    def test_renames_customer_id_with_space(self):
        df = pd.DataFrame({"Customer ID": [1, 2], "Monetary": [10, 20]})
        result = dash.normalize_columns(df)
        assert "CustomerID" in result.columns
        assert "Customer ID" not in result.columns

    def test_leaves_correct_column_name_untouched(self):
        df = pd.DataFrame({"CustomerID": [1, 2], "Monetary": [10, 20]})
        result = dash.normalize_columns(df)
        assert list(result.columns) == ["CustomerID", "Monetary"]

    def test_is_case_insensitive(self):
        df = pd.DataFrame({"customer id": [1, 2]})
        result = dash.normalize_columns(df)
        assert "CustomerID" in result.columns

    def test_does_not_touch_unrelated_columns(self):
        df = pd.DataFrame({"Recency": [1], "Frequency": [2], "Monetary": [3]})
        result = dash.normalize_columns(df)
        assert list(result.columns) == ["Recency", "Frequency", "Monetary"]

    def test_no_customerid_column_is_a_noop(self):
        df = pd.DataFrame({"Foo": [1], "Bar": [2]})
        result = dash.normalize_columns(df)
        assert list(result.columns) == ["Foo", "Bar"]


# --------------------------------------------------------------------------
# segment_color / segment_insight
# --------------------------------------------------------------------------

class TestSegmentLookups:
    @pytest.mark.parametrize("label", list(dash.COLORS.keys()))
    def test_known_segment_returns_mapped_color(self, label):
        assert dash.segment_color(label) == dash.COLORS[label]

    def test_unknown_segment_returns_default_color(self):
        assert dash.segment_color("Some New Cluster") == dash.DEFAULT_COLOR

    @pytest.mark.parametrize("label", list(dash.SEGMENT_INSIGHTS.keys()))
    def test_known_segment_returns_mapped_insight(self, label):
        assert dash.segment_insight(label) == dash.SEGMENT_INSIGHTS[label]

    def test_unknown_segment_returns_fallback_insight(self):
        assert dash.segment_insight("Some New Cluster") == dash.FALLBACK_INSIGHT


# --------------------------------------------------------------------------
# revenue_contribution_table
# --------------------------------------------------------------------------

class TestRevenueContributionTable:
    def _sample_rfm(self):
        return pd.DataFrame({
            "Segment": ["VIP Customers", "VIP Customers", "Lost Customers", "Regular Customers"],
            "Monetary": [300.0, 100.0, 50.0, 150.0],
        })

    def test_sums_revenue_per_segment(self):
        result = dash.revenue_contribution_table(self._sample_rfm())
        vip_row = result[result["Segment"] == "VIP Customers"].iloc[0]
        assert vip_row["Monetary"] == 400.0

    def test_sorted_descending_by_revenue(self):
        result = dash.revenue_contribution_table(self._sample_rfm())
        assert list(result["Monetary"]) == sorted(result["Monetary"], reverse=True)
        assert result.iloc[0]["Segment"] == "VIP Customers"

    def test_percentages_sum_to_100(self):
        result = dash.revenue_contribution_table(self._sample_rfm())
        assert result["Revenue %"].sum() == pytest.approx(100.0, abs=0.01)

    def test_single_segment_is_100_percent(self):
        df = pd.DataFrame({"Segment": ["VIP Customers"], "Monetary": [500.0]})
        result = dash.revenue_contribution_table(df)
        assert result.iloc[0]["Revenue %"] == 100.0


# --------------------------------------------------------------------------
# compute_cluster_to_label_map
# --------------------------------------------------------------------------

class _FakeScaler:
    """Identity scaler — returns whatever it's given, unscaled."""
    def transform(self, X):
        return np.asarray(X)


class _FakeKMeans:
    """
    Deterministic stand-in for a fitted KMeans model. Predicts cluster ids
    purely from the (already log-transformed) Recency value, so tests can
    control cluster assignment without needing a real fitted model.
    """
    def predict(self, X):
        # X columns are [log_recency, log_frequency, log_monetary]
        recency = X[:, 0]
        return np.where(recency < 2.0, 0, 1).astype(int)


class TestComputeClusterToLabelMap:
    def test_maps_cluster_to_majority_segment_label(self):
        # 3 low-recency customers labeled VIP -> cluster 0 should map to "VIP Customers"
        # 2 high-recency customers labeled Lost -> cluster 1 should map to "Lost Customers"
        rfm = pd.DataFrame({
            "Recency": [1, 1, 1, 300, 300],
            "Frequency": [10, 12, 9, 1, 1],
            "Monetary": [500, 600, 550, 20, 15],
            "Segment": [
                "VIP Customers", "VIP Customers", "VIP Customers",
                "Lost Customers", "Lost Customers",
            ],
        })
        mapping = dash.compute_cluster_to_label_map(rfm, _FakeKMeans(), _FakeScaler())
        assert mapping[0] == "VIP Customers"
        assert mapping[1] == "Lost Customers"

    def test_majority_vote_wins_within_a_noisy_cluster(self):
        # Cluster 0 (low recency) is mostly VIP with one mislabeled Regular —
        # majority vote should still pick VIP Customers.
        rfm = pd.DataFrame({
            "Recency": [1, 1, 1, 1],
            "Frequency": [10, 12, 9, 3],
            "Monetary": [500, 600, 550, 80],
            "Segment": ["VIP Customers", "VIP Customers", "VIP Customers", "Regular Customers"],
        })
        mapping = dash.compute_cluster_to_label_map(rfm, _FakeKMeans(), _FakeScaler())
        assert mapping[0] == "VIP Customers"

    def test_empty_cluster_falls_back_to_generic_label(self):
        # Only low-recency rows exist, so cluster 1 is never predicted and
        # should not appear in the mapping at all.
        rfm = pd.DataFrame({
            "Recency": [1, 1],
            "Frequency": [5, 6],
            "Monetary": [100, 120],
            "Segment": ["Regular Customers", "Regular Customers"],
        })
        mapping = dash.compute_cluster_to_label_map(rfm, _FakeKMeans(), _FakeScaler())
        assert 1 not in mapping
        assert mapping[0] == "Regular Customers"

    def test_mapping_is_robust_to_cluster_id_order(self):
        # Same data, but flip which physical cluster id represents "VIP" —
        # the mapping should follow the data, not a hardcoded index.
        class _FlippedKMeans:
            def predict(self, X):
                recency = X[:, 0]
                # VIPs (low recency) now get id 1 instead of 0
                return np.where(recency < 2.0, 1, 0).astype(int)

        rfm = pd.DataFrame({
            "Recency": [1, 1, 300, 300],
            "Frequency": [10, 12, 1, 1],
            "Monetary": [500, 600, 20, 15],
            "Segment": ["VIP Customers", "VIP Customers", "Lost Customers", "Lost Customers"],
        })
        mapping = dash.compute_cluster_to_label_map(rfm, _FlippedKMeans(), _FakeScaler())
        assert mapping[1] == "VIP Customers"
        assert mapping[0] == "Lost Customers"


# --------------------------------------------------------------------------
# REQUIRED_COLS / config sanity checks
# --------------------------------------------------------------------------

class TestConfigConsistency:
    def test_every_color_key_has_a_matching_insight(self):
        # Catches the case where someone adds a segment to COLORS but
        # forgets to add its SEGMENT_INSIGHTS entry (or vice versa) —
        # they'd silently fall back to generic styling/copy.
        assert set(dash.COLORS.keys()) == set(dash.SEGMENT_INSIGHTS.keys())

    def test_required_cols_are_all_strings(self):
        assert all(isinstance(c, str) for c in dash.REQUIRED_COLS)

    def test_silhouette_scores_cover_chosen_k(self):
        assert dash.CHOSEN_K in dash.SILHOUETTE_SCORES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
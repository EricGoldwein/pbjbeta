"""Focused data-quality guards for facility 315128 V3 dashboard APIs."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployments" / "pbj320-315128"


def _load_dashboard_module():
    entry = DEPLOY / "facility_315128_superdynamic_dashboard.py"
    if not entry.is_file():
        raise unittest.SkipTest(f"missing deployment entrypoint: {entry}")
    spec = importlib.util.spec_from_file_location("facility_315128_superdynamic_dashboard", entry)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DEPLOY))
    spec.loader.exec_module(mod)
    return mod


class Test315128DataQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_dashboard_module()
        cls.client = cls.mod.app.test_client()

    def test_summary_q4_2025_has_expected_scope(self):
        resp = self.client.get("/api/summary?quarter=2025Q4")
        self.assertEqual(resp.status_code, 200)
        summary = resp.get_json()
        self.assertEqual(summary.get("total_days"), 92)
        self.assertAlmostEqual(summary.get("avg_total_hprd"), 3.38, places=2)
        self.assertAlmostEqual(summary.get("avg_census"), 162.0, places=1)

    def test_analysis_bundle_matches_summary_for_q4(self):
        s = self.client.get("/api/summary?quarter=2025Q4").get_json()
        b = self.client.get("/api/analysis_bundle?quarter=2025Q4").get_json()
        bs = b.get("summary") or {}
        for key in ("avg_total_hprd", "avg_census", "total_days", "direct_rn_sub8"):
            self.assertEqual(bs.get(key), s.get(key), msg=key)

    def test_year_quarter_numeric_combo_not_empty(self):
        """Malformed year=2025&quarter=4 must not silently return zero-day scope."""
        bad = self.client.get("/api/summary?year=2025&quarter=4").get_json()
        self.assertNotEqual(bad.get("total_days"), 0)

    def test_missing_quarter_filter_does_not_fabricate_zero_hprd(self):
        empty = self.client.get("/api/summary?quarter=2099Q1").get_json()
        self.assertEqual(empty.get("total_days"), 0)
        self.assertIsNone(empty.get("avg_total_hprd"))
        self.assertIsNone(empty.get("total_rn_sub8"))
        self.assertIsNone(empty.get("direct_rn_sub8"))

    def test_case_mix_geo_bundle_marks_unavailable_geo(self):
        s = self.client.get("/api/summary?quarter=2025Q4").get_json()
        bundle = (s or {}).get("case_mix_geo_bundle") or {}
        self.assertFalse(bundle.get("available"))
        self.assertIn("No geo CMI rows", bundle.get("note", ""))


if __name__ == "__main__":
    unittest.main()

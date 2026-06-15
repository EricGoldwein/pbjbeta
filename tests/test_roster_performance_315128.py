"""Roster API performance and correctness guards for facility 315128."""
from __future__ import annotations

import importlib.util
import sys
import time
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


class Test315128RosterPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_dashboard_module()
        cls.client = cls.mod.app.test_client()

    def test_nursing_roster_day_view_under_budget(self):
        # Init + EIN pre-warm runs in setUpClass; prime summary cache once.
        self.client.get("/api/ein-nursing-employees?quarter=2025Q4")
        t0 = time.perf_counter()
        resp = self.client.get(
            "/api/ein-nursing-employees?quarter=2025Q4&work_date=2025-11-01"
        )
        elapsed = time.perf_counter() - t0
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json() or {}
        self.assertTrue(data.get("available"))
        self.assertEqual(data.get("source"), "precomputed")
        self.assertTrue(data.get("work_day_filtered"))
        employees = data.get("employees") or []
        self.assertGreater(len(employees), 0)
        self.assertLess(elapsed, 3.0, msg=f"roster took {elapsed:.2f}s")
        for row in employees:
            self.assertIsNotNone(row.get("hours_selected_work_day"))

    def test_warm_roster_day_view_under_budget(self):
        self.client.get("/api/ein-nursing-employees?quarter=2025Q4&work_date=2025-11-01")
        t0 = time.perf_counter()
        resp = self.client.get(
            "/api/ein-nursing-employees?quarter=2025Q4&work_date=2025-11-01"
        )
        elapsed = time.perf_counter() - t0
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 3.0, msg=f"warm roster took {elapsed:.2f}s")

    def test_missing_work_day_not_zero_employees_when_unfiltered(self):
        resp = self.client.get(
            "/api/ein-nursing-employees?quarter=2099Q1&work_date=2099-01-01"
        )
        data = resp.get_json() or {}
        self.assertTrue(data.get("available"))
        self.assertEqual(data.get("total"), 0)
        self.assertEqual(data.get("employees"), [])

    def test_quarter_year_combo_still_works(self):
        resp = self.client.get("/api/summary?year=2025&quarter=4")
        self.assertEqual(resp.status_code, 200)
        summary = resp.get_json() or {}
        self.assertNotEqual(summary.get("total_days"), 0)


if __name__ == "__main__":
    unittest.main()

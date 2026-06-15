"""Invariant tests for PBJ320 shared metric calculation helpers."""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from pbj_metric_helpers import (
    coerce_num,
    compliance_share,
    contract_share_pct,
    pooled_hprd_from_df,
    pooled_hprd_from_series,
    round_half_up_display,
    row_sum_hours,
    safe_divide,
    TOTAL_NURSE_HOUR_COLS,
    DIRECT_CARE_HOUR_COLS,
)


class TestCoerceAndSafeDivide(unittest.TestCase):
    def test_missing_stays_none(self):
        self.assertIsNone(coerce_num(None))
        self.assertIsNone(coerce_num(np.nan))
        self.assertIsNone(coerce_num(""))
        self.assertIsNone(safe_divide(10, None))
        self.assertIsNone(safe_divide(None, 5))

    def test_real_zero_preserved(self):
        self.assertEqual(coerce_num(0), 0.0)
        self.assertEqual(safe_divide(0, 10), 0.0)

    def test_zero_denominator_returns_none(self):
        self.assertIsNone(safe_divide(5, 0))


class TestPooledHprd(unittest.TestCase):
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "MDScensus": [100, 100, 0, np.nan],
                "Hrs_RN": [8.0, 4.0, 2.0, 8.0],
                "Hrs_LPN": [4.0, 2.0, 1.0, 4.0],
                "Hrs_CNA": [8.0, 4.0, 2.0, 8.0],
                "Hrs_NAtrn": [0.0, 0.0, 0.0, 0.0],
                "Hrs_MedAide": [0.0, 0.0, 0.0, 0.0],
            }
        )

    def test_pooled_not_average_of_daily_rates(self):
        df = pd.DataFrame(
            {
                "MDScensus": [100, 50],
                "Hrs_RN": [8.0, 2.0],
                "Hrs_LPN": [4.0, 1.0],
                "Hrs_CNA": [8.0, 2.0],
                "Hrs_NAtrn": [0.0, 0.0],
                "Hrs_MedAide": [0.0, 0.0],
            }
        )
        direct = pooled_hprd_from_df(df, DIRECT_CARE_HOUR_COLS)
        # pooled: (20+5)/(100+50) = 25/150; avg of daily rates: (0.2+0.1)/2 = 0.15
        self.assertAlmostEqual(direct or 0, 25 / 150, places=4)
        avg_daily = (20 / 100 + 5 / 50) / 2
        self.assertNotAlmostEqual(direct or 0, avg_daily, places=3)

    def test_missing_hours_excluded_not_zero(self):
        df = pd.DataFrame(
            {
                "MDScensus": [100, 100],
                "Hrs_RN": [8.0, np.nan],
                "Hrs_LPN": [4.0, 4.0],
                "Hrs_CNA": [8.0, 8.0],
                "Hrs_NAtrn": [0.0, 0.0],
                "Hrs_MedAide": [0.0, 0.0],
            }
        )
        val = pooled_hprd_from_df(df, DIRECT_CARE_HOUR_COLS)
        self.assertAlmostEqual(val or 0, 0.20, places=4)

    def test_empty_returns_none(self):
        self.assertIsNone(pooled_hprd_from_df(pd.DataFrame(), TOTAL_NURSE_HOUR_COLS))

    def test_pooled_from_series(self):
        hrs = pd.Series([10.0, 20.0, np.nan])
        cen = pd.Series([5.0, 5.0, 5.0])
        self.assertAlmostEqual(pooled_hprd_from_series(hrs, cen) or 0, 3.0, places=4)


class TestContractAndCompliance(unittest.TestCase):
    def test_contract_share_missing_denominator(self):
        self.assertIsNone(contract_share_pct(5, None))
        self.assertIsNone(contract_share_pct(5, 0))

    def test_compliance_no_observed_days(self):
        self.assertIsNone(compliance_share(0, 0))
        self.assertAlmostEqual(compliance_share(8, 10) or 0, 80.0, places=4)


class TestRoundHalfUpDisplay(unittest.TestCase):
    def test_nan_returns_none_not_zero(self):
        self.assertIsNone(round_half_up_display(None))
        self.assertIsNone(round_half_up_display(np.nan))

    def test_half_up(self):
        self.assertEqual(round_half_up_display(2.345, 2), 2.35)


class TestRowSumHours(unittest.TestCase):
    def test_partial_missing_row_is_nan(self):
        df = pd.DataFrame({"Hrs_RN": [1.0, np.nan], "Hrs_LPN": [2.0, 2.0]})
        s = row_sum_hours(df, ("Hrs_RN", "Hrs_LPN"))
        self.assertFalse(math.isnan(s.iloc[0]))
        self.assertTrue(math.isnan(s.iloc[1]))


class TestFacilityReportLibNullPolicy(unittest.TestCase):
    def test_round_half_up_missing_returns_none(self):
        from facility_report_lib import round_half_up

        self.assertIsNone(round_half_up(None))
        self.assertIsNone(round_half_up(np.nan))


if __name__ == "__main__":
    unittest.main()

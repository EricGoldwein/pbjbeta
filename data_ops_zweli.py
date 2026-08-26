"""ZWELI CHECK — semantic plausibility layer (separate from structural validation).

Named for the historical false story: administrative staffing appeared to drop
~20% due to a unit/scaling corruption (e.g. hours vs minutes / ~60×), not a
real operational change. Schema-valid files can still lie.

States: PASS | REQUIRES_REVIEW | BLOCKED | NOT_RUN
Never reduce to a single boolean — store findings.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence


class ZweliState(str, Enum):
    PASS = "PASS"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class ZweliSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class BaselineAvailability(str, Enum):
    """How to interpret a missing Provider Info Zweli baseline."""

    PRESENT = "present"
    UNAVAILABLE_IN_RUNTIME = "unavailable_in_runtime"
    NONE_EXPECTED = "none_expected"  # genuine first / no comparable predecessor


@dataclass
class ZweliFinding:
    check_id: str
    source_id: str
    metric_field: str
    current_release: Optional[str]
    baseline_release: Optional[str]
    observed_value: Any
    observed_change: Any
    expected_or_baseline: Any
    severity: ZweliSeverity
    explanation: str
    sample_evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class ZweliReport:
    source_id: str
    release_id: Optional[str]
    profile: str
    state: ZweliState
    findings: list[ZweliFinding] = field(default_factory=list)
    checked_at: str = ""
    baseline_release: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "release_id": self.release_id,
            "profile": self.profile,
            "state": self.state.value,
            "findings": [f.to_dict() for f in self.findings],
            "checked_at": self.checked_at,
            "baseline_release": self.baseline_release,
            "notes": self.notes,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def aggregate_state(findings: Sequence[ZweliFinding]) -> ZweliState:
    if not findings:
        return ZweliState.PASS
    if any(f.severity == ZweliSeverity.BLOCKING for f in findings):
        return ZweliState.BLOCKED
    if any(f.severity == ZweliSeverity.WARNING for f in findings):
        return ZweliState.REQUIRES_REVIEW
    return ZweliState.PASS


# Scale factors historically associated with unit corruption (hours↔minutes, etc.).
DEFAULT_SCALE_SIGNATURES = (60.0, 1.0 / 60.0, 100.0, 1.0 / 100.0)


def detect_scale_signature(
    current: float,
    baseline: float,
    *,
    signatures: Sequence[float] = DEFAULT_SCALE_SIGNATURES,
    rel_tol: float = 0.08,
) -> Optional[float]:
    """Return matching scale factor if current/baseline ≈ signature, else None."""
    if baseline == 0 or current == 0:
        return None
    ratio = current / baseline
    for sig in signatures:
        if math.isclose(ratio, sig, rel_tol=rel_tol, abs_tol=0):
            return sig
    return None


def check_scale_shift(
    *,
    source_id: str,
    metric_field: str,
    current_release: str,
    baseline_release: str,
    current_value: float,
    baseline_value: float,
    check_id: str = "unit_scale_signature",
) -> Optional[ZweliFinding]:
    sig = detect_scale_signature(current_value, baseline_value)
    if sig is None:
        return None
    return ZweliFinding(
        check_id=check_id,
        source_id=source_id,
        metric_field=metric_field,
        current_release=current_release,
        baseline_release=baseline_release,
        observed_value=current_value,
        observed_change=current_value / baseline_value if baseline_value else None,
        expected_or_baseline=baseline_value,
        severity=ZweliSeverity.BLOCKING,
        explanation=(
            f"Population metric shows ~{sig:g}× shift vs baseline — classic unit/"
            f"scaling corruption signature (Zweli). Not treated as a real event."
        ),
        sample_evidence={"ratio": current_value / baseline_value, "matched_signature": sig},
    )


def check_impossible_nonpositive_count(
    *,
    source_id: str,
    metric_field: str,
    release_id: str,
    value: float,
) -> Optional[ZweliFinding]:
    if value is not None and value <= 0:
        return ZweliFinding(
            check_id="invariant_nonpositive_count",
            source_id=source_id,
            metric_field=metric_field,
            current_release=release_id,
            baseline_release=None,
            observed_value=value,
            observed_change=None,
            expected_or_baseline="> 0",
            severity=ZweliSeverity.BLOCKING,
            explanation=f"{metric_field} must be positive; got {value}",
        )
    return None


@dataclass
class ProviderInfoMetrics:
    release_id: str
    row_count: int
    unique_ccn: int
    null_rates: dict[str, float] = field(default_factory=dict)
    schema_columns: tuple[str, ...] = ()


def compare_provider_info_releases(
    current: ProviderInfoMetrics,
    baseline: Optional[ProviderInfoMetrics],
    *,
    source_id: str = "cms.provider_info",
    row_review_ratio: float = 0.08,
    row_block_ratio: float = 0.35,
    null_review_delta: float = 0.15,
    baseline_availability: BaselineAvailability | str = BaselineAvailability.PRESENT,
    expected_baseline_release: str | None = None,
) -> ZweliReport:
    """Provider Information Zweli profile v0 — distribution / schema / nulls.

    Does NOT use naive 'change > 20% = bad'. Large shifts → REQUIRES_REVIEW
    unless a scale signature or invariant fires → BLOCKED.

    Baseline semantics:
    - baseline present → normal PASS / REQUIRES_REVIEW / BLOCKED
    - expected prior unavailable in this runtime → NOT_RUN (not REQUIRES_REVIEW)
    - genuine first / no prior expected → REQUIRES_REVIEW (no_baseline)
    """
    if isinstance(baseline_availability, str):
        baseline_availability = BaselineAvailability(baseline_availability)

    findings: list[ZweliFinding] = []
    findings.append(
        ZweliFinding(
            check_id="row_count_present",
            source_id=source_id,
            metric_field="row_count",
            current_release=current.release_id,
            baseline_release=baseline.release_id if baseline else None,
            observed_value=current.row_count,
            observed_change=None,
            expected_or_baseline="> 0",
            severity=ZweliSeverity.INFO,
            explanation="Current release row count",
        )
    )
    bad = check_impossible_nonpositive_count(
        source_id=source_id,
        metric_field="row_count",
        release_id=current.release_id,
        value=float(current.row_count),
    )
    if bad:
        findings.append(bad)

    if baseline is None:
        if baseline_availability == BaselineAvailability.PRESENT:
            # Inconsistent args: claimed present but no metrics — treat as runtime gap.
            baseline_availability = BaselineAvailability.UNAVAILABLE_IN_RUNTIME

        if baseline_availability == BaselineAvailability.UNAVAILABLE_IN_RUNTIME:
            findings.append(
                ZweliFinding(
                    check_id="baseline_unavailable_in_runtime",
                    source_id=source_id,
                    metric_field="baseline",
                    current_release=current.release_id,
                    baseline_release=expected_baseline_release,
                    observed_value=None,
                    observed_change=None,
                    expected_or_baseline=expected_baseline_release,
                    severity=ZweliSeverity.INFO,
                    explanation=(
                        "Comparable prior release is expected but not available in this "
                        "runtime — Zweli NOT_RUN (runtime availability), not a substantive "
                        "quality review finding."
                    ),
                    sample_evidence={
                        "expected_baseline_release": expected_baseline_release,
                        "reason": "runtime_availability",
                    },
                )
            )
            return ZweliReport(
                source_id=source_id,
                release_id=current.release_id,
                profile="provider_info_v0",
                state=ZweliState.NOT_RUN,
                findings=findings,
                checked_at=_utc_now(),
                baseline_release=expected_baseline_release,
                notes="Baseline unavailable in this runtime",
            )

        # Genuine first / no comparable predecessor expected
        findings.append(
            ZweliFinding(
                check_id="no_baseline",
                source_id=source_id,
                metric_field="baseline",
                current_release=current.release_id,
                baseline_release=None,
                observed_value=None,
                observed_change=None,
                expected_or_baseline=None,
                severity=ZweliSeverity.WARNING,
                explanation=(
                    "No prior comparable release expected/exists — REQUIRES_REVIEW "
                    "for genuine first release"
                ),
            )
        )
        return ZweliReport(
            source_id=source_id,
            release_id=current.release_id,
            profile="provider_info_v0",
            state=aggregate_state(findings),
            findings=findings,
            checked_at=_utc_now(),
            notes="Genuine first release — no prior comparable baseline",
        )

    # Schema drift
    cur_cols = set(current.schema_columns)
    base_cols = set(baseline.schema_columns)
    if cur_cols != base_cols:
        added = sorted(cur_cols - base_cols)
        removed = sorted(base_cols - cur_cols)
        findings.append(
            ZweliFinding(
                check_id="schema_change",
                source_id=source_id,
                metric_field="schema_columns",
                current_release=current.release_id,
                baseline_release=baseline.release_id,
                observed_value={"added": added, "removed": removed},
                observed_change=None,
                expected_or_baseline=list(baseline.schema_columns),
                severity=ZweliSeverity.WARNING,
                explanation="Schema columns changed vs baseline",
                sample_evidence={"added": added, "removed": removed},
            )
        )

    # Row / provider count deltas
    if baseline.row_count > 0:
        delta = abs(current.row_count - baseline.row_count) / baseline.row_count
        scale = check_scale_shift(
            source_id=source_id,
            metric_field="row_count",
            current_release=current.release_id,
            baseline_release=baseline.release_id,
            current_value=float(current.row_count),
            baseline_value=float(baseline.row_count),
        )
        if scale:
            findings.append(scale)
        elif delta >= row_block_ratio:
            findings.append(
                ZweliFinding(
                    check_id="row_count_extreme_shift",
                    source_id=source_id,
                    metric_field="row_count",
                    current_release=current.release_id,
                    baseline_release=baseline.release_id,
                    observed_value=current.row_count,
                    observed_change=delta,
                    expected_or_baseline=baseline.row_count,
                    severity=ZweliSeverity.WARNING,
                    explanation=(
                        f"Row count shifted {delta:.1%} vs baseline — large but not a "
                        f"known unit signature; REQUIRES_REVIEW (not auto-blocked)."
                    ),
                )
            )
        elif delta >= row_review_ratio:
            findings.append(
                ZweliFinding(
                    check_id="row_count_notable_shift",
                    source_id=source_id,
                    metric_field="row_count",
                    current_release=current.release_id,
                    baseline_release=baseline.release_id,
                    observed_value=current.row_count,
                    observed_change=delta,
                    expected_or_baseline=baseline.row_count,
                    severity=ZweliSeverity.WARNING,
                    explanation=f"Row count shifted {delta:.1%} vs baseline",
                )
            )

    if baseline.unique_ccn > 0:
        ccn_delta = abs(current.unique_ccn - baseline.unique_ccn) / baseline.unique_ccn
        scale = check_scale_shift(
            source_id=source_id,
            metric_field="unique_ccn",
            current_release=current.release_id,
            baseline_release=baseline.release_id,
            current_value=float(current.unique_ccn),
            baseline_value=float(baseline.unique_ccn),
        )
        if scale:
            findings.append(scale)
        elif ccn_delta >= row_review_ratio:
            findings.append(
                ZweliFinding(
                    check_id="ccn_count_shift",
                    source_id=source_id,
                    metric_field="unique_ccn",
                    current_release=current.release_id,
                    baseline_release=baseline.release_id,
                    observed_value=current.unique_ccn,
                    observed_change=ccn_delta,
                    expected_or_baseline=baseline.unique_ccn,
                    severity=ZweliSeverity.WARNING,
                    explanation=f"Unique CCN count shifted {ccn_delta:.1%}",
                )
            )

    # Null-rate changes for important fields
    for field_name, cur_rate in current.null_rates.items():
        base_rate = baseline.null_rates.get(field_name)
        if base_rate is None:
            continue
        delta = abs(cur_rate - base_rate)
        if delta >= null_review_delta:
            findings.append(
                ZweliFinding(
                    check_id="null_rate_shift",
                    source_id=source_id,
                    metric_field=field_name,
                    current_release=current.release_id,
                    baseline_release=baseline.release_id,
                    observed_value=cur_rate,
                    observed_change=delta,
                    expected_or_baseline=base_rate,
                    severity=ZweliSeverity.WARNING,
                    explanation=f"Null rate for {field_name} changed by {delta:.1%}",
                )
            )

    return ZweliReport(
        source_id=source_id,
        release_id=current.release_id,
        profile="provider_info_v0",
        state=aggregate_state(findings),
        findings=findings,
        checked_at=_utc_now(),
        baseline_release=baseline.release_id,
    )


def metrics_from_provider_csv(path: Path, release_id: str) -> ProviderInfoMetrics:
    """Lightweight metrics for Zweli (not a full ETL)."""
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = tuple(reader.fieldnames or [])
        rows = list(reader)
    n = len(rows)
    ccn_key = None
    for cand in ("CMS Certification Number (CCN)", "PROVNUM", "ccn"):
        if cand in cols:
            ccn_key = cand
            break
    ccns = set()
    null_rates: dict[str, float] = {}
    important = [
        c
        for c in cols
        if c
        in {
            "Provider Name",
            "Provider Address",
            "City/Town",
            "State",
            "Special Focus Status",
            "Overall Rating",
            "Processing Date",
        }
    ]
    for col in important:
        nulls = sum(1 for r in rows if not str(r.get(col, "")).strip())
        null_rates[col] = (nulls / n) if n else 1.0
    if ccn_key:
        for r in rows:
            v = str(r.get(ccn_key, "")).strip()
            if v:
                ccns.add(v)
    return ProviderInfoMetrics(
        release_id=release_id,
        row_count=n,
        unique_ccn=len(ccns),
        null_rates=null_rates,
        schema_columns=cols,
    )


def expected_prior_month(year: int, month: int) -> tuple[int, int]:
    """Previous calendar month for monthly Provider Info cadence."""
    if month <= 1:
        return year - 1, 12
    return year, month - 1


def run_provider_info_zweli(
    current_csv: Path,
    current_release: str,
    baseline_csv: Path | None = None,
    baseline_release: str | None = None,
    *,
    baseline_availability: BaselineAvailability | str | None = None,
    expected_baseline_release: str | None = None,
) -> ZweliReport:
    current = metrics_from_provider_csv(current_csv, current_release)
    baseline = None
    if baseline_csv is not None and baseline_csv.is_file():
        baseline = metrics_from_provider_csv(
            baseline_csv, baseline_release or baseline_csv.stem
        )
        availability = BaselineAvailability.PRESENT
    elif baseline_availability is not None:
        availability = (
            BaselineAvailability(baseline_availability)
            if isinstance(baseline_availability, str)
            else baseline_availability
        )
    else:
        # Default when caller omits both baseline file and availability:
        # treat as expected-prior unavailable (runtime), not genuine first.
        availability = BaselineAvailability.UNAVAILABLE_IN_RUNTIME
    return compare_provider_info_releases(
        current,
        baseline,
        baseline_availability=availability,
        expected_baseline_release=expected_baseline_release or baseline_release,
    )


def write_zweli_report(report: ZweliReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def not_run_report(source_id: str, reason: str = "Profile not implemented") -> ZweliReport:
    return ZweliReport(
        source_id=source_id,
        release_id=None,
        profile="none",
        state=ZweliState.NOT_RUN,
        findings=[],
        checked_at=_utc_now(),
        notes=reason,
    )

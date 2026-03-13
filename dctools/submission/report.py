"""Validation report formatting for model submissions.

Generates human-readable and machine-readable reports summarizing
whether a submitted dataset meets the Data Challenge specifications.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckStatus(str, Enum):
    """Status of a single validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Full validation report for a model submission.

    Attributes
    ----------
    model_name : str
        Human-readable name of the submitted model.
    data_path : str
        Path to the submitted dataset.
    checks : list[CheckResult]
        Ordered list of validation checks performed.
    overall_pass : bool
        ``True`` when no check has status FAIL.
    """

    model_name: str
    data_path: str
    checks: List[CheckResult] = field(default_factory=list)
    overall_pass: bool = True

    # -- helpers --------------------------------------------------
    def add(self, result: CheckResult) -> None:
        """Append a check result and update *overall_pass*."""
        self.checks.append(result)
        if result.status == CheckStatus.FAIL:
            self.overall_pass = False

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def n_skip(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.SKIP)

    # -- serialization --------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "model_name": self.model_name,
            "data_path": self.data_path,
            "overall_pass": self.overall_pass,
            "summary": {
                "total": len(self.checks),
                "pass": self.n_pass,
                "fail": self.n_fail,
                "warn": self.n_warn,
                "skip": self.n_skip,
            },
            "checks": [asdict(c) for c in self.checks],
        }

    def save_json(self, path: str | Path) -> None:
        """Write the report as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    # -- pretty print ---------------------------------------------
    def pretty(self, *, use_color: bool = True) -> str:
        """Return a human-readable multi-line summary.

        Inspired by WeatherBench-style CLI output: compact, coloured, and
        immediately actionable for participants.
        """
        _ICONS = {
            CheckStatus.PASS: ("✅", "\033[92m"),   # green
            CheckStatus.FAIL: ("❌", "\033[91m"),   # red
            CheckStatus.WARN: ("⚠️ ", "\033[93m"),  # yellow
            CheckStatus.SKIP: ("⏭️ ", "\033[90m"),  # grey
        }
        RESET = "\033[0m"

        lines: List[str] = []
        separator = "═" * 72

        lines.append(f"\n╔{separator}╗")
        lines.append(f"║{'  SUBMISSION VALIDATION REPORT':^72}║")
        lines.append(f"╠{separator}╣")
        lines.append(f"║  Model : {self.model_name:<61}║")
        # Truncate data path to fit in box
        _dp = str(self.data_path)
        if len(_dp) > 58:
            _dp = "..." + _dp[-55:]
        lines.append(f"║  Data  : {_dp:<61}║")
        lines.append(f"╠{separator}╣")

        for check in self.checks:
            icon, color_code = _ICONS.get(check.status, ("?", ""))
            if use_color:
                status_str = f"{color_code}{icon} {check.status.value.upper():4s}{RESET}"
            else:
                status_str = f"{icon} {check.status.value.upper():4s}"
            line = f"║  {status_str}  {check.name:<50}"
            # Pad to fit box (accounting for ANSI codes not being visible)
            lines.append(line)
            if check.message and check.status in (CheckStatus.FAIL, CheckStatus.WARN):
                msg = check.message
                # Wrap long messages
                while len(msg) > 64:
                    lines.append(f"║         {msg[:64]}")
                    msg = msg[64:]
                lines.append(f"║         {msg}")

        lines.append(f"╠{separator}╣")
        summary_line = (
            f"  RESULT: {self.n_pass} passed, {self.n_fail} failed, "
            f"{self.n_warn} warnings, {self.n_skip} skipped"
        )
        if self.overall_pass:
            result_str = "✅ SUBMISSION READY" if use_color else "PASS — SUBMISSION READY"
        else:
            result_str = "❌ SUBMISSION NOT READY" if use_color else "FAIL — SUBMISSION NOT READY"

        lines.append(f"║{summary_line:<72}║")
        lines.append(f"║  {'':>18}{result_str:^34}{'':>18}║")
        lines.append(f"╚{separator}╝\n")

        return "\n".join(lines)

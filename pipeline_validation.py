"""Collect per-stage validation results for one pipeline run.

This report is separate from exam JSON: validation details describe the run
and must not change the exam schema or its content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PipelineValidationReport:
    """A small JSON-serializable record of checks, issues, and output files."""

    input_path: str
    output_dir: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    outcome: str = "running"
    exit_code: int | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        check: str,
        status: str,
        message: str,
        *,
        details: Any = None,
    ) -> None:
        """Append one check result while keeping optional evidence structured."""
        entry: dict[str, Any] = {
            "stage": stage,
            "check": check,
            "status": status,
            "message": message,
        }
        if details is not None:
            entry["details"] = details
        self.checks.append(entry)

    def add_artifact(self, name: str, path: Path) -> None:
        """Remember where the pipeline wrote an inspectable stage output."""
        self.artifacts[name] = str(path)

    def finish(self, exit_code: int) -> None:
        """Set the final run status using the pipeline's established exit codes."""
        self.exit_code = exit_code
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if exit_code == 1:
            self.outcome = "failed"
        elif exit_code == 2 or any(
            check["status"] in {"failed", "warning"} for check in self.checks
        ):
            self.outcome = "completed_with_issues"
        else:
            self.outcome = "completed"

    def to_dict(self) -> dict[str, Any]:
        """Return the stable report shape written to the output folder."""
        return {
            "report_version": "1.0",
            "input": self.input_path,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "checks": self.checks,
            "artifacts": self.artifacts,
        }

"""JSON reporter — serializes AnalysisResult to the canonical JSON schema."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from stockgrader.models import AnalysisResult


class JSONReporter:
    """Produce the canonical machine-readable JSON output (spec §13.1)."""

    def render(self, result: AnalysisResult) -> str:
        """Return a pretty-printed JSON string."""
        return json.dumps(result.to_json_dict(), indent=2, default=str)

    def save(
        self,
        result:   AnalysisResult,
        runs_dir: str | Path = "runs",
        filename: str | None = None,
    ) -> Path:
        """
        Write JSON to runs/TICKER_YYYY-MM-DD_<run_id>.json.
        Returns the path that was written.
        """
        base = Path(runs_dir)
        base.mkdir(parents=True, exist_ok=True)

        if filename is None:
            date_str = result.as_of.strftime("%Y-%m-%d")
            run_id   = result.run_id or uuid.uuid4().hex[:8]
            filename = f"{result.ticker}_{date_str}_{run_id}.json"

        path = base / filename
        path.write_text(self.render(result), encoding="utf-8")
        return path

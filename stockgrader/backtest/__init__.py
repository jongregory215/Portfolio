"""
Backtesting engine — point-in-time grade validation, walk-forward portfolio
backtest, and weight calibration. Requires a PIT data subscription
(Sharadar/Nasdaq Data Link). Dependency-isolated from the live tool.
"""
from .pit_adapter import (
    PITAdapter,
    BacktestContaminationError,
    require_pit,
    YFinancePITAdapter,
    SharadarAdapter,
    SyntheticPITAdapter,
)
from .grade_validator import validate_grades, GradeValidationReport
from .walk_forward import walk_forward_from_scores, BacktestResult, CostModel
from .calibrator import calibrate_weights, CalibrationResult
from .report import (
    render_validation_report,
    render_backtest_report,
    render_calibration_report,
    save_report,
    to_json,
)

__all__ = [
    "PITAdapter", "BacktestContaminationError", "require_pit",
    "YFinancePITAdapter", "SharadarAdapter", "SyntheticPITAdapter",
    "validate_grades", "GradeValidationReport",
    "walk_forward_from_scores", "BacktestResult", "CostModel",
    "calibrate_weights", "CalibrationResult",
    "render_validation_report", "render_backtest_report",
    "render_calibration_report", "save_report", "to_json",
]

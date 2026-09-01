"""Shared configuration for ASCENTRA Module A."""

from pathlib import Path


RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ascentra_student_data_refined.csv"
ARTIFACT_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_CANDIDATES = (
    "academic_risk",
    "Target",
    "target",
    "risk",
    "risk_label",
)

REMOVED_FEATURES = (
    "quiz_average",
    "late_submission_count",
    "lms_resource_access",
    "assignment_average",
    "lms_login_frequency",
)

BASE_NUMERIC_FEATURES = (
    "semester",
    "attendance_percentage",
    "ca1_score",
    "ca2_score",
    "ca3_score",
    "best_2_ca_average",
    "mid_term_score",
    "previous_semester_tgpa",
)

CATEGORICAL_FEATURES = ("academic_trend",)

ENGINEERED_NUMERIC_FEATURES = (
    "ca_average",
    "ca_improvement",
    "ca_consistency",
    "attendance_best2_interaction",
    "attendance_midterm_interaction",
    "tgpa_best2_interaction",
    "previous_tgpa_available",
)

BASE_INPUT_FEATURES = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPECTED_TRENDS = ("Declining", "Stable", "Improving")

RISK_THRESHOLDS = {
    "low_max": 0.35,
    "medium_max": 0.70,
}

GEMINI_MODEL = "gemini-2.5-flash"

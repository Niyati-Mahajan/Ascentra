"""Representative ASCENTRA Module A student-profile tests."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .predict import academic_risk_prediction
except ImportError:
    from predict import academic_risk_prediction


REPRESENTATIVE_PROFILES = [
    {
        "student_id": "LOW_RISK_STUDENT",
        "semester": 4,
        "attendance_percentage": 92.0,
        "ca1_score": 25.0,
        "ca2_score": 27.0,
        "ca3_score": 26.0,
        "best_2_ca_average": 26.5,
        "mid_term_score": 26.0,
        "previous_semester_tgpa": 8.6,
        "academic_trend": "Improving",
    },
    {
        "student_id": "MEDIUM_RISK_STUDENT",
        "semester": 3,
        "attendance_percentage": 72.0,
        "ca1_score": 15.0,
        "ca2_score": 18.0,
        "ca3_score": 13.0,
        "best_2_ca_average": 16.5,
        "mid_term_score": 15.0,
        "previous_semester_tgpa": 6.3,
        "academic_trend": "Stable",
    },
    {
        "student_id": "HIGH_RISK_STUDENT",
        "semester": 5,
        "attendance_percentage": 52.0,
        "ca1_score": 7.0,
        "ca2_score": 10.0,
        "ca3_score": 9.0,
        "best_2_ca_average": 9.5,
        "mid_term_score": 8.0,
        "previous_semester_tgpa": 4.9,
        "academic_trend": "Declining",
    },
    {
        "student_id": "SEMESTER_1_STUDENT",
        "semester": 1,
        "attendance_percentage": 68.0,
        "ca1_score": 14.0,
        "ca2_score": 16.0,
        "ca3_score": 13.0,
        "best_2_ca_average": 15.0,
        "mid_term_score": 14.0,
        "previous_semester_tgpa": None,
        "academic_trend": "Stable",
    },
    {
        "student_id": "SENIOR_SEMESTER_STUDENT",
        "semester": 7,
        "attendance_percentage": 61.0,
        "ca1_score": 12.0,
        "ca2_score": 14.0,
        "ca3_score": 15.0,
        "best_2_ca_average": 14.5,
        "mid_term_score": 13.0,
        "previous_semester_tgpa": 5.4,
        "academic_trend": "Declining",
    },
]


def run_representative_profiles(output_path: Path | None = None) -> list[dict]:
    results = [
        {
            "input": profile,
            "prediction": academic_risk_prediction(profile),
        }
        for profile in REPRESENTATIVE_PROFILES
    ]
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "reports" / "representative_profile_results.json"
    print(json.dumps(run_representative_profiles(output), indent=2))

"""Factor-linked Academic Risk Agent for ASCENTRA Module A."""

from __future__ import annotations

import math
from typing import Any


def risk_level_from_probability(probability: float, thresholds: dict[str, float]) -> str:
    if probability < thresholds["low_max"]:
        return "LOW"
    if probability < thresholds["medium_max"]:
        return "MEDIUM"
    return "HIGH"


def decision_from_level(level: str) -> str:
    return {
        "LOW": "MONITOR",
        "MEDIUM": "ADVISE",
        "HIGH": "ESCALATE",
    }[level]


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float) and math.isnan(value):
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def recommendation_for_factor(feature: str, value: Any) -> dict[str, str]:
    display_value = _format_value(value)
    base_feature = feature.split("=")[0]

    if base_feature == "attendance_percentage":
        if isinstance(value, (int, float)) and value >= 75:
            return {
                "risk_factor": f"Attendance = {display_value}%",
                "why_it_matters": "The model assigns a risk-increasing contribution, but the raw attendance value is currently healthy.",
                "recommendation": "Maintain attendance above 75% and continue routine attendance monitoring.",
            }
        return {
            "risk_factor": f"Attendance = {display_value}%",
            "why_it_matters": "Low or weakening attendance is contributing to the predicted academic risk.",
            "recommendation": "Weekly attendance monitoring, advisor meeting, and a documented attendance improvement target.",
        }
    if base_feature in {"ca1_score", "ca2_score", "ca3_score", "best_2_ca_average", "ca_average"}:
        if isinstance(value, (int, float)) and value >= 20:
            return {
                "risk_factor": f"{feature} = {display_value}/30",
                "why_it_matters": "The model assigns a risk-increasing contribution, but the raw assessment score is strong.",
                "recommendation": "Maintain current assessment performance and review only if later CA scores decline.",
            }
        return {
            "risk_factor": f"{feature} = {display_value}/30",
            "why_it_matters": "Continuous assessment performance is pushing the prediction toward academic risk.",
            "recommendation": "Targeted tutoring/revision sessions and weekly assessment monitoring.",
        }
    if base_feature == "ca_improvement":
        return {
            "risk_factor": f"CA improvement = {display_value}",
            "why_it_matters": "The change from CA1 to CA3 is contributing to the predicted academic risk.",
            "recommendation": "Review the CA sequence with the student and set a short-term improvement target for the next assessment.",
        }
    if base_feature == "ca_consistency":
        return {
            "risk_factor": f"CA consistency variation = {display_value}",
            "why_it_matters": "Variation across CA scores suggests unstable assessment performance.",
            "recommendation": "Use weekly practice checks to stabilize performance across upcoming assessments.",
        }
    if base_feature in {"attendance_best2_interaction", "attendance_midterm_interaction"}:
        return {
            "risk_factor": f"{feature} = {display_value}",
            "why_it_matters": "The combination of attendance and academic performance is contributing to predicted risk.",
            "recommendation": "Coordinate attendance monitoring with targeted tutoring so study support and class participation improve together.",
        }
    if base_feature == "tgpa_best2_interaction":
        return {
            "risk_factor": f"TGPA and CA performance interaction = {display_value}",
            "why_it_matters": "Prior semester performance combined with current CA performance is contributing to predicted risk.",
            "recommendation": "Create a recovery plan that connects prior weak areas with current assessment preparation.",
        }
    if base_feature == "previous_tgpa_available":
        return {
            "risk_factor": f"Previous TGPA available = {display_value}",
            "why_it_matters": "The model is using whether previous-semester TGPA exists as semester context, especially for Semester 1 students.",
            "recommendation": "For Semester 1 students, rely on current attendance, CA scores, mid-term score, and trend rather than prior TGPA.",
        }
    if base_feature == "mid_term_score":
        if isinstance(value, (int, float)) and value >= 20:
            return {
                "risk_factor": f"Mid-term = {display_value}/30",
                "why_it_matters": "The model assigns a risk-increasing contribution, but the raw mid-term score is strong.",
                "recommendation": "Maintain current revision practices and monitor the next assessment result.",
            }
        return {
            "risk_factor": f"Mid-term = {display_value}/30",
            "why_it_matters": "Mid-term performance is contributing to the predicted academic risk.",
            "recommendation": "Identify weak subjects and provide targeted revision support before the next assessment.",
        }
    if base_feature == "previous_semester_tgpa":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return {
                "risk_factor": "Previous semester TGPA = unavailable",
                "why_it_matters": "The student is in Semester 1, so previous TGPA is not available and is not interpreted as poor performance.",
                "recommendation": "Use current attendance, CA scores, mid-term score, and academic trend for early-semester monitoring.",
            }
        return {
            "risk_factor": f"Previous Semester TGPA = {display_value}",
            "why_it_matters": "Prior academic performance is contributing to the predicted academic risk.",
            "recommendation": (
                "Maintain the current academic plan and review previous-semester weak subjects if TGPA is below 6.5."
                if isinstance(value, (int, float)) and value >= 6.5
                else "Review previous-semester weak subjects and create a semester recovery plan."
            ),
        }
    if base_feature == "academic_trend":
        return {
            "risk_factor": f"Academic trend = {display_value}",
            "why_it_matters": "The student's academic trajectory is pushing the model toward risk.",
            "recommendation": "Schedule advisor follow-up and agree on short-cycle performance checkpoints.",
        }
    if base_feature == "semester":
        return {
            "risk_factor": f"Semester = {display_value}",
            "why_it_matters": "Semester is an ordered academic progression feature and may affect risk context.",
            "recommendation": "Review semester-specific academic workload and support requirements.",
        }
    return {
        "risk_factor": f"{feature} = {display_value}",
        "why_it_matters": "This factor is contributing to the model's risk prediction.",
        "recommendation": "Review this factor with the advisor and define a measurable improvement target.",
    }


def build_academic_risk_agent_response(
    prediction: dict[str, Any],
    top_risk_factors: list[dict[str, Any]],
) -> dict[str, Any]:
    interventions = [
        recommendation_for_factor(factor["feature"], factor.get("value"))
        for factor in top_risk_factors
    ]

    level = prediction["risk_level"]
    immediate_actions = []
    if level == "HIGH":
        immediate_actions.append("Advisor escalation within one week.")
    elif level == "MEDIUM":
        immediate_actions.append("Advisor check-in and early support plan.")
    else:
        immediate_actions.append("Continue routine academic monitoring.")

    return {
        "immediate_actions": immediate_actions,
        "priority_interventions": interventions[:3],
        "student_support_strategy": (
            "Prioritize the strongest SHAP-ranked risk factors and convert each into a measurable support action."
        ),
        "advisor_follow_up": "Review progress after the next attendance cycle and assessment checkpoint.",
        "monitoring_metrics": [
            "attendance_percentage",
            "best_2_ca_average",
            "mid_term_score",
            "academic_trend",
            "previous_semester_tgpa when available",
        ],
        "escalation_condition": "Escalate if risk remains HIGH or the top risk factors worsen at the next review.",
    }

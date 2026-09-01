from app.ai.career_guide import CareerGuideAIService
from app.ai.domain_scope import BORDERLINE, IN_SCOPE, OUT_OF_SCOPE, classify_domain_scope
from app.config import settings


def test_readiness_is_in_scope():
    decision = classify_domain_scope("What is my readiness?")
    assert decision.scope == IN_SCOPE


def test_dsa_for_placements_is_in_scope():
    decision = classify_domain_scope("How should I prepare DSA for placements?")
    assert decision.scope == IN_SCOPE


def test_linked_lists_are_borderline_without_placement_context():
    decision = classify_domain_scope("Explain linked lists")
    assert decision.scope == BORDERLINE
    assert "placement" in decision.instruction


def test_reverse_linked_list_code_is_redirected_without_model_call(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    service = CareerGuideAIService()
    calls = []

    def fake_request(model, payload):
        calls.append((model, payload))
        return "should not be used"

    monkeypatch.setattr(service, "_request_model", fake_request)
    result = service.generate_response(
        {"skill_profile": {"target_gaps": ["DSA"]}},
        "Give me JavaScript code to reverse a linked list",
        [],
    )
    assert calls == []
    assert result["model_used"] == "scope_guardrail"
    assert "Linked lists are relevant to DSA interview preparation" in result["answer"]
    assert "JavaScript" not in result["answer"]


def test_two_sum_solution_request_is_redirected_without_solution():
    decision = classify_domain_scope("Help me solve Two Sum")
    assert decision.scope == OUT_OF_SCOPE


def test_linked_list_practice_for_interviews_is_allowed():
    decision = classify_domain_scope("Which linked-list problems should I practice for interviews?")
    assert decision.scope == IN_SCOPE


def test_assignment_request_is_out_of_scope():
    decision = classify_domain_scope("Write my DBMS assignment")
    assert decision.scope == OUT_OF_SCOPE


def test_sql_for_backend_placements_is_allowed():
    decision = classify_domain_scope("What SQL concepts matter for backend placements?")
    assert decision.scope == IN_SCOPE


def test_joke_is_redirected():
    decision = classify_domain_scope("Tell me a joke")
    assert decision.scope == OUT_OF_SCOPE


def test_follow_up_uses_recent_placement_context():
    history = [{"role": "user", "content": "Which DSA topics should I prepare for placements?"}]
    decision = classify_domain_scope("What about linked lists?", history)
    assert decision.scope == IN_SCOPE

from src.qwen_math_final import (
    audit_python,
    extract_integer,
    normalize_integer,
    terminal_boxed_integer,
    vote_candidates,
)


def test_integer_normalization_preserves_large_values():
    value = "9" * 5000
    assert normalize_integer(value) == value


def test_extraction_precedence():
    assert extract_integer("work 17\n\\boxed{42}") == ("42", "boxed")
    assert extract_integer("Final answer: -8") == ("-8", "final_marker")


def test_terminal_boxed_requires_clean_tail():
    assert terminal_boxed_integer("proof\n\\boxed{-12}.") == "-12"
    assert terminal_boxed_integer("\\boxed{7} more text") is None


def test_vote_tie_uses_earliest_sample():
    candidates = [
        {"sample_index": 0, "answer": "2"},
        {"sample_index": 1, "answer": "3"},
        {"sample_index": 2, "answer": "3"},
        {"sample_index": 3, "answer": "2"},
    ]
    result = vote_candidates(candidates)
    assert result["answer"] == "2"
    assert result["tie"] is True


def test_pal_static_audit():
    assert audit_python("from fractions import Fraction\nprint(3)") == (True, "ok")
    allowed, reason = audit_python("import os\nos.system('id')")
    assert not allowed
    assert reason == "import_denied"

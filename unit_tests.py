import pytest
from pipeline import (
    classify_complaint,
    analyze_complaint_with_openai,
    analyze_complaint_with_litellm,
    analyze_complaint_with_huggingface,
    ComplaintInfo
)

@pytest.fixture
def sample_complaint():
    return "I’ve been trying to reach your customer support for a week with no response!"

def test_analyze_complaint_with_openai(sample_complaint):
    result = analyze_complaint_with_openai(sample_complaint)
    assert result is None or isinstance(result, ComplaintInfo)

def test_analyze_complaint_with_litellm(sample_complaint):
    result = analyze_complaint_with_litellm(sample_complaint)
    assert result is None or isinstance(result, ComplaintInfo)

def test_analyze_complaint_with_huggingface(sample_complaint):
    result = analyze_complaint_with_huggingface(sample_complaint)
    assert result is None or isinstance(result, ComplaintInfo)

def test_classify_complaint(sample_complaint):
    result = classify_complaint(sample_complaint)
    assert result is None or isinstance(result, ComplaintInfo)

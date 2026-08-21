"""
LLM-as-judge regression suite for taxonomy tagging quality.

Given a company description and a candidate taxonomy tag (produced by an
LLM classifier), the judge scores whether the tag is correct, and if not,
classifies *why* it's wrong (wrong parent category, overweighted on one
feature, missing a secondary tag, or genuinely ambiguous and needing human
review).

This mirrors real taxonomy-QA work: at scale, "is this tag right" is rarely
binary — most errors are near-misses (right domain, wrong specificity) that
a keyword/exact-match check can't catch, which is why this uses a judge
instead of simple string comparison against `expected_taxonomy`.

Usage:
    export OPENAI_API_KEY=...
    python judge.py dataset/cases.jsonl
"""

import json
import os
import sys
from dataclasses import dataclass

JUDGE_PROMPT = """You are auditing a company-taxonomy tagging system used to \
classify B2B companies for an intent-data platform.

Company description:
{company_description}

Candidate taxonomy tag assigned by the system:
{candidate_taxonomy}

Evaluate whether the candidate tag correctly and specifically categorizes \
this company. A tag can be wrong in several distinct ways:
- WRONG_PARENT: the top-level category itself is incorrect
- OVERWEIGHTED_FEATURE: it latched onto one feature (e.g. "offers financing") \
  instead of the company's core business
- MISSING_SECONDARY: the primary category is right but a secondary tag is \
  missing and the company meaningfully spans two categories
- AMBIGUOUS: the description genuinely supports multiple readings and this \
  should be flagged for human review, not auto-resolved
- CORRECT: the tag is right

Respond in strict JSON:
{{"verdict": "<CORRECT|WRONG_PARENT|OVERWEIGHTED_FEATURE|MISSING_SECONDARY|AMBIGUOUS>",
  "reasoning": "<one or two sentences>"}}
"""


@dataclass
class CaseResult:
    case_id: str
    verdict: str
    expected_label: str
    reasoning: str
    match: bool


def call_judge(company_description: str, candidate_taxonomy: str) -> dict:
    """Calls an LLM judge. Swap in your provider of choice."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = JUDGE_PROMPT.format(
        company_description=company_description,
        candidate_taxonomy=candidate_taxonomy,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def label_to_verdict(label: str) -> str:
    """Maps the dataset's ground-truth label to the judge's verdict vocabulary."""
    mapping = {
        "correct": "CORRECT",
        "incorrect_wrong_parent_category": "WRONG_PARENT",
        "incorrect_overweighted_on_financing_feature": "OVERWEIGHTED_FEATURE",
        "incorrect_missing_secondary_category": "MISSING_SECONDARY",
        "ambiguous_flag_for_review": "AMBIGUOUS",
    }
    return mapping.get(label, "UNKNOWN")


def run(dataset_path: str) -> None:
    results = []
    with open(dataset_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    for case in cases:
        verdict = call_judge(case["company_description"], case["candidate_taxonomy"])
        expected_verdict = label_to_verdict(case["label"])
        match = verdict["verdict"] == expected_verdict
        results.append(
            CaseResult(
                case_id=case["id"],
                verdict=verdict["verdict"],
                expected_label=expected_verdict,
                reasoning=verdict["reasoning"],
                match=match,
            )
        )

    passed = sum(r.match for r in results)
    print(f"\n{passed}/{len(results)} cases matched expected verdict\n")
    for r in results:
        status = "PASS" if r.match else "FAIL"
        print(f"[{status}] {r.case_id}: judge said {r.verdict} (expected {r.expected_label})")
        if not r.match:
            print(f"        reasoning: {r.reasoning}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/cases.jsonl"
    run(path)

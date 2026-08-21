
A collection of roduct/AI projects, focused on the evaluation
layer of LLM systems datasets, rubrics, and LLM-as-judge setups.
## Projects


### [taxonomy-tagging-llm-judge](./taxonomy-tagging-llm-judge)
An LLM-as-judge eval suite for B2B company-taxonomy classification, built
from real quality problems encountered managing taxonomy tagging at scale
(1000+ taxonomies, 600+ companies) at ET Signal. The judge classifies *why*
a tag is wrong (wrong parent category, overweighted on one feature, missing
a secondary tag, or genuinely ambiguous) rather than a binary pass/fail.

### [loan-eligibility-agent-evals](./loan-eligibility-agent-evals)
A promptfoo-based regression suite for an agentic loan-eligibility
assistant. Covers happy-path, incomplete-info, and adversarial cases
(attempts to bypass checks or extract internal thresholds), scored with a
combination of deterministic guardrails and a rubric-based LLM judge.

## About

Uday Jain — Product Analyst at Times Internet (ET Signal), IIT Delhi.

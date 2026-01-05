# XML Output Format Specification

**Competition**: Kaggle Google Tunix Hackathon  
**Version**: 1.0  
**Last Updated**: January 2026

---

## Overview

This document specifies the required XML output format for JudicAIta model responses. The format ensures that reasoning traces and final answers are clearly structured for evaluation.

## Required Format

All model outputs **MUST** follow this exact XML structure:

```xml
<reasoning>
[Step-by-step reasoning trace with detailed analysis]
</reasoning>
<answer>
[Final answer or conclusion]
</answer>
```

## Tag Specifications

### `<reasoning>` Tag

**Required**: Yes

**Purpose**: Contains the complete reasoning trace showing how the model arrives at its conclusion.

**Requirements**:
- Must be present in every response
- Must contain substantive reasoning content
- Target minimum: **100 tokens** of detailed reasoning
- Should include logical steps, analysis, and supporting arguments
- May include legal citations, precedents, or relevant principles

**Example**:
```xml
<reasoning>
The primary legal issue is whether a verbal contract for the sale of real estate is enforceable.

Step 1: Identify applicable law. The Statute of Frauds (e.g., Cal. Civ. Code § 1624) requires certain contracts to be in writing.

Step 2: Determine if this transaction is covered. Real property transactions fall within the Statute of Frauds.

Step 3: Apply the rule. Since no written instrument exists for this real estate sale, the contract lacks the required formality.

Therefore, the verbal agreement is likely unenforceable due to the Statute of Frauds requirement.
</reasoning>
```

### `<answer>` Tag

**Required**: Yes

**Purpose**: Contains the final answer or conclusion.

**Requirements**:
- Must be present in every response
- Must directly answer the query
- Should be concise and clear
- Should follow logically from the reasoning

**Example**:
```xml
<answer>
No, a verbal contract for the sale of land is generally unenforceable. Under the Statute of Frauds, contracts for the sale of real property must be in writing to be enforceable.
</answer>
```

## Complete Example

**Query**: "Is a verbal contract to sell a house enforceable?"

**Expected Output**:
```xml
<reasoning>
The question concerns the enforceability of a verbal (oral) contract for the sale of real property.

Step 1: The Statute of Frauds is the governing legal principle. This statute, adopted in various forms across jurisdictions, requires certain contracts to be evidenced by a signed writing.

Step 2: Real estate transactions are specifically enumerated under the Statute of Frauds. This includes sales, leases for more than one year, and mortgages.

Step 3: Applying this rule to a verbal house sale agreement: since no written contract exists, the agreement fails to meet the statutory requirement.

Step 4: While there are limited exceptions (such as part performance or promissory estoppel in some jurisdictions), the general rule is that verbal real estate contracts are unenforceable.
</reasoning>
<answer>
No, a verbal contract to sell a house is generally not enforceable. The Statute of Frauds requires contracts for the sale of real property to be in writing and signed by the party to be charged. Without a written agreement, the contract is voidable and cannot be enforced in court.
</answer>
```

## Validation Criteria

### Format Validation

The following regex patterns are used to validate output format:

```python
import re

def validate_xml_format(text: str) -> dict:
    """Validate XML format compliance."""
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    
    return {
        "valid": has_reasoning and has_answer,
        "has_reasoning": has_reasoning,
        "has_answer": has_answer
    }
```

### Content Extraction

```python
def extract_xml_content(text: str) -> tuple[str | None, str | None]:
    """Extract reasoning and answer from XML tags."""
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    
    return reasoning, answer
```

### Quality Metrics

| Metric | Minimum | Target |
|--------|---------|--------|
| Format Compliance | 80% | 100% |
| Reasoning Tokens | 100 | 150+ |
| Answer Presence | 100% | 100% |
| Logical Coherence | 0.5 | 0.7+ |

## Token Limits

Per competition requirements:

- **Maximum Output Tokens**: ≤1000 tokens (recommended: 512)
- **Reasoning Section**: Target 100-500 tokens
- **Answer Section**: Target 50-200 tokens

## Competition Compliance

This format aligns with the Kaggle Google Tunix Hackathon requirements:

1. **Model Output**: Follows required XML format ✅
2. **Reasoning Trace**: Contains detailed step-by-step reasoning ✅
3. **Final Answer**: Clearly separated from reasoning ✅
4. **Token Limit**: Configured for max 512 tokens (under 1K limit) ✅

## Related Documentation

- [GRPO Training Guide](./GRPO_TRAINING.md)
- [Colab Validation Guide](./COLAB_VALIDATION_GUIDE.md)
- [Hackathon Submission Checklist](./HACKATHON_SUBMISSION_CHECKLIST.md)

---

## Changelog

- **v1.0** (January 2026): Initial specification

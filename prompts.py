READABILITY_PROMPT = """\
You are a Senior Code Reviewer evaluating ONLY code readability.
Do NOT judge functionality, correctness, or algorithm efficiency (time or space complexity).

Based on the context of the problem description to evaluate the code readability based on these criteria:
- Overall readability (naming clarity, formatting, indentation, line length)
- Maintainability (structure, logical grouping, avoiding duplication)
- Language leverage (using built-ins and features effectively)
- Control flow clarity (avoid deep nesting, unclear logic)

Scoring (1-5):
- 5: Excellent
- 4: Good
- 3: Average
- 2: Poor
- 1: Very poor

Output format:
{{
    "score": <integer 1-5>
}}

Evaluate the readability of the following {lang} code:

**Problem description:**
{description}

**Code snippet:**
```
{code}
```
"""

CORRECTNESS_PROMPT = """\
You are a Senior Code Reviewer evaluating ONLY the logical correctness of the code.

DO NOT judge: algorithm efficiency, readability or syntax.

You must analyse whether the code would produce correct outputs
for ALL possible valid inputs of the problem.

In other words, determine if the implementation would pass in every case,
not just some example cases.

Then evaluate the code on the scale from 1 to 5, based on your analysis,
where 5 means the code would pass in all cases.

*Output format (JSON ONLY)*:
{{
  "score": <integer 1-5>
}}

Evaluate the correctness of the following code snippet:

**Problem description:**
{description}

**Code snippet:**
```
{code}
```
"""

EFFICIENCY_PROMPT = """\
You are a Senior Code Reviewer. Your SOLE purpose is to evaluate the Time and Space Complexity of the code 
from an algorithmic optimization perspective. Do not judge style or bugs.

Even if the code appears logically wrong or produces incorrect output,
you MUST ignore correctness and only evaluate the algorithmic approach
and its theoretical efficiency.

You must analyse whether the algorithmic approach used in the code is
optimal or near-optimal for solving the described problem.

If the algorithm in the code is clear:
- Score based on whether it uses the most optimal known algorithm.

If the code is incomplete, logically incorrect, or partially implemented:
- Try to infer the intended algorithmic approach from the code.
- Evaluate based on that inferred approach.

If the approach is unclear or cannot be inferred:
- Assign a low score.

Score on a scale from 1 to 5:
1 = very poor algorithmic approach or unclear  
5 = optimal algorithm (matches theoretical best known complexity)

**Output format (JSON ONLY)**:
{{
    "score": <integer 1-5>
}}

Evaluate the efficiency of the following code snippet:

**Problem Description**:
{description}

**Code Snippet**:
```
{code}
```
"""

def format_readability_prompt(row):
    return READABILITY_PROMPT.format(
        lang=row['lang'],
        description=row['description'],
        code=row['code'],
    )

def format_correctness_prompt(row):
    return CORRECTNESS_PROMPT.format(
        description=row['description'],
        code=row['code'],
    )

def format_efficiency_prompt(row):
    return EFFICIENCY_PROMPT.format(
        description=row['description'],
        time_limit=row['time_limit'],
        memory_limit=row['memory_limit'],
        code=row['code']
    )

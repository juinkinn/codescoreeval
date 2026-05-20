SYNTAX_PROMPT = """\
You are a Senior Code Reviewer evaluating ONLY code readability, syntax validity, maintainability, and effective use of language features.
Do NOT judge functionality, correctness, or algorithm efficiency.
Consider the context of competitive programming (CP), where code is written under time pressure but still needs to be debuggable and modifiable.

Evaluate the code based on these criteria:
- Syntax validity (must be correct for the language)
- Readability (naming clarity, formatting, indentation, line length)
- Maintainability (structure, logical grouping, avoiding duplication)
- Language leverage (using built-ins and features effectively)
- Control flow clarity (avoid deep nesting, unclear logic)

Scoring (1-5):
- 5: Excellent
- 4: Good
- 3: Average
- 2: Poor
- 1: Very poor

Evaluate the syntax/readability of the following {lang} code:

{code}

Final answer (1, 2, 3, 4, or 5 only):
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

Evaluate the correctness of the following code snippet:

*Problem description:*
{description}

*Code snippet:*
```
{code}
```

Final answer (1, 2, 3, 4, or 5 only):
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

Evaluate the efficiency of the following code snippet:

**Problem Description**:
{description}

**Code Snippet**:

```
{code}
```

Final answer (1, 2, 3, 4, or 5 only):
"""

READABILITY_PAIRWISE = """\
You are a Senior Code Reviewer comparing readability of TWO code submissions.

Do NOT consider functionality, correctness, or algorithm efficiency (time or space complexity).

Based on the context of the problem description to evaluate the code readability based on these criteria:
- Overall readability (naming clarity, formatting, indentation, line length)
- Maintainability (structure, logical grouping, avoiding duplication)
- Language leverage (using built-ins and features effectively)
- Control flow clarity (avoid deep nesting, unclear logic)

Compare TWO code submissions based ONLY on code readability.

Problem Description:
{description}

Code A ({lang1}):
{code1}

Code B ({lang2}):
{code2}

Which code is better?

Choose A or B unless they are strictly equivalent with respect to the criterion. 
Choose "Both" only if there is no meaningful difference.

Final answer (A, B, or Both only): """

CORRECTNESS_PAIRWISE = """\
You are a Senior Code Reviewer comparing correctness of TWO code submissions.

DO NOT consider: algorithm efficiency, readability or syntax.

You must analyse whether the code would produce correct outputs
for ALL possible valid inputs of the problem.

In other words, determine if the implementation would pass in every case,
not just some example cases.

Compare TWO code submissions based ONLY on code correctness.

Problem Description:
{description}

Code A ({lang1}):
{code2}

Code B ({lang2}):
{code1}

Which code is better?

Choose A or B unless they are strictly equivalent with respect to the criterion. 
Choose "Both" only if there is no meaningful difference.

Final answer (A, B, or Both only): """

EFFICIENCY_PAIRWISE = """\
You are a Senior Code Reviewer comparing the Time and Space Complexity of TWO code submissions from an algorithmic optimization perspective.

DO NOT consider: correctness, bug, readability or syntax.

Even if the code appears logically wrong or produces incorrect output,
you MUST ignore correctness and only evaluate the algorithmic approach
and its theoretical efficiency.

You must analyse whether the algorithmic approach used in the code is
optimal or near-optimal for solving the described problem.

If the algorithm in the code is clear:
- Compare based on whether it uses the most optimal known algorithm.

If the code is incomplete, logically incorrect, or partially implemented:
- Try to infer the intended algorithmic approach from the code.
- Compare based on that inferred approach.

Compare TWO code submissions based ONLY on code algorithm efficiency.

Problem Description:
{description}

Code A ({lang1}):
{code1}

Code B ({lang2}):
{code2}

Which code is better?

Choose A or B unless they are strictly equivalent with respect to the criterion. 
Choose "Both" only if there is no meaningful difference.

Final answer (A, B, or Both only): """

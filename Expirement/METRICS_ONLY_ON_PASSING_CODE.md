# Why Metrics Are Only Computed on Passing Code

## The Problem

When a model generates **non-functional code** (examples, documentation, or broken code), computing code quality metrics on it produces **misleading data**.

### Example Scenario:

**Model generates examples instead of implementation:**
```python
# What the model outputs:
has_close_elements([1.0, 2.0, 3.0], 0.5) # returns False
has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) # returns True
```

**This is valid Python syntax** (function calls), so metrics get computed:
- **LOC:** 2 lines
- **Cyclomatic Complexity:** 0 (no control flow)
- **Halstead Effort:** ~150
- **Cognitive Complexity:** 0

**But this is misleading!** The model didn't write an implementation - it wrote examples.

---

## The Solution

**Metrics are now only computed on code that passes all tests.**

### Implementation:

```python
# OLD (computed on everything):
if compute_metrics and attempt1_code.strip():
    attempt1_metrics = analyze_code_metrics(code_for_metrics)

# NEW (only on passing code):
if compute_metrics and attempt1_code.strip() and attempt1_passed is True:
    attempt1_metrics = analyze_code_metrics(code_for_metrics)
```

---

## What This Means for Your Data

### Before (Misleading):

```csv
model,task_id,pass_k,LOC,CC,cognitive,halstead_effort
mistral:7b,HumanEval/0,0.0,2,0,0.0,150.5
```

**Problem:** 
- pass_k = 0.0 (failed)
- But LOC/CC/Halstead show "metrics" for non-functional code
- **Misleading:** Looks like model generated code with low complexity

### After (Accurate):

```csv
model,task_id,pass_k,LOC,CC,cognitive,halstead_effort
mistral:7b,HumanEval/0,0.0,,,
```

**Clear:**
- pass_k = 0.0 (failed)
- Metrics are empty/null (no functional code to measure)
- **Accurate:** Model failed to generate working code

---

## Why This Is Better for Research

### Your Research Questions:

1. **Does RAG improve code generation quality?**
2. **Which models generate better code?**
3. **What's the code complexity of successful solutions?**

### With Metrics on All Code (Bad):

```csv
Baseline (no RAG):
- Pass@k: 0.0 (failed)
- LOC: 2, CC: 0 (measuring examples)
- ❌ Misleading: Suggests simple but functional code

With RAG:
- Pass@k: 1.0 (passed!)
- LOC: 15, CC: 5 (measuring actual implementation)
- ❌ Confusing: Why is "better" code more complex?
```

### With Metrics on Passing Code Only (Good):

```csv
Baseline (no RAG):
- Pass@k: 0.0 (failed)
- LOC: null, CC: null (no functional code)
- ✅ Clear: Model failed completely

With RAG:
- Pass@k: 1.0 (passed!)
- LOC: 15, CC: 5 (measuring actual implementation)
- ✅ Clear: RAG enabled successful code generation
```

---

## What Gets Measured

### Code Quality Metrics Are For:
✅ **Working solutions** - Code that passes all tests  
✅ **Comparing successful approaches** - Which working solution is simpler/better?  
✅ **Understanding complexity** - How complex are the solutions that work?

### Code Quality Metrics Are NOT For:
❌ **Failed attempts** - Examples, documentation, broken code  
❌ **Syntax errors** - Can't parse, metrics meaningless  
❌ **Wrong implementations** - Measuring the wrong thing

---

## Research Implications

### What You Can Conclude:

#### Scenario 1: Model Improves with RAG
```
Without RAG:
- Pass@10 = 0.2 (2 out of 10 passed)
- Avg LOC (of passing): 12
- Avg CC (of passing): 4

With RAG:
- Pass@10 = 0.8 (8 out of 10 passed)
- Avg LOC (of passing): 14
- Avg CC (of passing): 5
```

**Conclusion:** 
- RAG significantly improves success rate (0.2 → 0.8)
- Passing solutions are slightly more complex (CC 4 → 5)
- This makes sense: RAG helps generate correct, complete solutions

#### Scenario 2: Code-Specific Models Work Better
```
General Model (mistral:7b):
- Pass@10 = 0.1
- Metrics: Mostly empty (few passing attempts)

Code Model (codellama:7b):
- Pass@10 = 0.7
- Avg LOC: 18, CC: 6
- More complex but working solutions
```

**Conclusion:**
- Code models succeed more often
- Their solutions are more elaborate (higher complexity)
- This is expected for functional code vs. simple examples

---

## How to Interpret Empty Metrics

### In Your Excel/CSV Output:

```csv
model,task,n,c,pass_k,attempt1_CC,attempt2_CC
mistral:7b,HE/0,2,0,0.0,,,
mistral:7b,HE/7,2,1,1.0,3.0,
codellama:7b,HE/0,2,2,1.0,5.0,4.0
```

**Row 1:** Both attempts failed → No metrics (empty)  
**Row 2:** Attempt 1 passed (CC=3.0), Attempt 2 failed (empty)  
**Row 3:** Both passed → Both have metrics

### Statistical Analysis:

When computing averages, handle nulls appropriately:

```python
# Only average non-null metrics (passing attempts)
avg_cc = df[df['attempt1_CC'].notna()]['attempt1_CC'].mean()

# Or count how many attempts have metrics
success_with_metrics = df['attempt1_CC'].notna().sum()
```

---

## What This Doesn't Change

### Pass@k Calculation:
✅ **Still works correctly** - Counts all attempts (pass and fail)
```python
n = 2  # Total attempts (both count)
c = 1  # Correct attempts (only passing)
Pass@2 = estimate_pass_at_k(n=2, c=1, k=2) = 1.0
```

### Logging:
✅ **All attempts still logged** - experiment_logs.jsonl contains everything
- Failed attempts: pass_1=0, metrics={}
- Passed attempts: pass_1=1, metrics={LOC:15, CC:5, ...}

### Test Execution:
✅ **All attempts still tested** - Every generated code is evaluated
- Just metrics aren't computed on failures

---

## Benefits

### 1. Data Integrity
✅ Metrics only measure what they're designed to measure (working code)

### 2. Clear Interpretation
✅ Empty metrics = failed attempt  
✅ Populated metrics = successful attempt

### 3. Research Validity
✅ Comparing apples to apples (working solutions vs working solutions)  
✅ Not comparing working code to examples/documentation

### 4. Statistical Soundness
✅ Averages are meaningful (avg of working solutions)  
✅ Not polluted by "fake code" metrics

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Metrics on failing code** | ✅ Computed (misleading) | ❌ Skipped (clear) |
| **Metrics on passing code** | ✅ Computed | ✅ Computed |
| **Pass@k calculation** | ✅ Correct | ✅ Correct |
| **Data interpretation** | ❌ Confusing | ✅ Clear |
| **Research validity** | ❌ Questionable | ✅ Sound |

---

## FAQ

### Q: Why not compute metrics and just flag them?
**A:** Flagging creates confusion in analysis. Empty/null is clearer than "3.0 (but invalid)".

### Q: What if I want metrics on everything?
**A:** The raw code is still logged in `experiment_logs.jsonl`. You can post-process it separately if needed.

### Q: Will this affect my Pass@k scores?
**A:** No! Pass@k only cares about pass/fail status, not metrics.

### Q: How do I calculate "average complexity"?
**A:** Average only the non-null metrics:
```python
df[df['CC'].notna()]['CC'].mean()  # Average of passing attempts only
```

### Q: What if no attempts pass?
**A:** All metrics will be null, and Pass@k will be 0.0. This clearly shows complete failure.

---

## Example Research Findings

With this approach, you can make clear statements like:

✅ **"Model X achieved Pass@10 = 0.6, with successful solutions averaging 12 LOC and CC of 4.2"**

❌ NOT: "Model X generated code with 5 LOC and CC of 1.5" (when it actually failed and just wrote examples)

✅ **"RAG improved Pass@10 from 0.2 to 0.8, though successful solutions were slightly more complex (CC: 3.5 → 4.8)"**

❌ NOT: "RAG made code more complex" (confusing working code with examples)

---

Your research will now have **clear, interpretable, scientifically sound metrics**! 🎯

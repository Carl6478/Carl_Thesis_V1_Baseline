# How to Trace Pass@k Parameters in Your Code

This guide shows you exactly where to look in the code and output files to trace the Pass@k formula parameters (n, c, k).

---

## Quick Reference: Where Parameters Are Used

| Parameter | Meaning | Where to Find |
|-----------|---------|---------------|
| **n** | Total number of attempts | Lines 725, 761, 1011 |
| **c** | Number of correct attempts | Lines 726-732, 761, 1012 |
| **k** | Samples to evaluate | Lines 744, 761, 1023 |
| **Pass@k** | Calculated result | Lines 744, 763, 1023 |

---

## 1. Tracing in Batch Evaluation

### Code Location: Lines 723-744

```python
# Compute aggregated metrics using Pass@k formula
# Count total attempts (n) and correct attempts (c)
n = 2  # Total number of attempts  ← LINE 725: n is set
c = 0  # Number of correct attempts  ← LINE 726: c initialized

# Count correct attempts
if attempt1_passed is True:  ← LINE 729: c gets incremented
    c += 1
if attempt2_passed is True:  ← LINE 731: c gets incremented
    c += 1

# Pass@1: just the first attempt
pass_1 = 1 if attempt1_passed else (0 if attempt1_passed is False else None)

# Pass@k using the proper formula
if attempt1_passed is None and attempt2_passed is None:
    pass_k_2 = None
else:
    # ← LINE 744: Formula is called with n, c, k
    pass_k_2 = estimate_pass_at_k(n=n, c=c, k=2)
```

### How to Add Debug Prints:

Add after line 732:
```python
print(f"DEBUG [{task_id}] n={n}, c={c}, k=2")
print(f"  attempt1_passed={attempt1_passed}, attempt2_passed={attempt2_passed}")
```

Add after line 744:
```python
print(f"  Pass@2 = {pass_k_2:.4f}")
```

### What Gets Saved to CSV/Excel (Line 761):

```python
row = {
    # ... other fields ...
    "n_total_attempts": n,      # ← The n parameter
    "c_correct_attempts": c,    # ← The c parameter
    "k": 2,                     # ← The k parameter (always 2 for batch)
    "pass_k_2": pass_k_2,      # ← The calculated Pass@k result
    # ...
}
```

### Example Excel Output:

| model | task_id | n_total_attempts | c_correct_attempts | k | pass_k_2 | attempt1_passed | attempt2_passed |
|-------|---------|------------------|--------------------|----|----------|-----------------|-----------------|
| codellama:7b | HumanEval/7 | 2 | 0 | 2 | 0.0000 | 0 | 0 |
| codellama:7b | HumanEval/21 | 2 | 1 | 2 | 1.0000 | 1 | 0 |
| mistral:7b | HumanEval/7 | 2 | 2 | 2 | 1.0000 | 1 | 1 |

**How to read this:**
- Row 1: n=2 attempts, c=0 correct → Pass@2 = 0.0 (both failed)
- Row 2: n=2 attempts, c=1 correct → Pass@2 = 1.0 (one passed, guaranteed)
- Row 3: n=2 attempts, c=2 correct → Pass@2 = 1.0 (both passed)

---

## 2. Tracing in Single-Run Pass@k

### Code Location: Lines 1010-1023

```python
# Compute pass@k using the proper formula
n = pass_at_k_value  # Total number of attempts  ← LINE 1011: n from UI input
c = len(passed_attempts)  # Number of correct attempts  ← LINE 1012: c counted

# Pass@1: just the first attempt
pass_1 = 1 if attempts[0]["passed"] else (0 if attempts[0]["passed"] is False else None)

# Pass@k using the estimate_pass_at_k function
all_none = all(att["passed"] is None for att in attempts)
if all_none:
    pass_at_k = None
else:
    pass_at_k = estimate_pass_at_k(n=n, c=c, k=pass_at_k_value)  ← LINE 1023: Formula called
```

### How to Add Debug Prints:

Add after line 1012:
```python
print(f"DEBUG: n={n}, c={c}, k={pass_at_k_value}")
print(f"  passed_attempts={passed_attempts}")
print(f"  all attempts: {[(i+1, att['passed']) for i, att in enumerate(attempts)]}")
```

Add after line 1023:
```python
print(f"  Pass@{pass_at_k_value} = {pass_at_k:.4f}")
```

### UI Display (Lines 1026-1037):

```python
st.markdown("## 📊 Pass@k Results")
# Shows the formula with actual values:
st.info(f"Using Pass@k formula: Pass@k = 1 - C(n-c, k) / C(n, k) where n={n}, c={c}, k={pass_at_k_value}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Pass@1", f"{pass_1}" if pass_1 is not None else "N/A")
with col2:
    if pass_at_k is None:
        st.metric(f"Pass@{pass_at_k_value}", "N/A")
    else:
        st.metric(f"Pass@{pass_at_k_value}", f"{pass_at_k:.4f}")  # Shows decimal result
with col3:
    st.metric("Passed Attempts (c)", f"{c}/{n}")  # Shows c and n
with col4:
    st.metric("Total Attempts (n)", f"{n}")  # Shows n
```

### Example UI Display:

```
Using Pass@k formula: Pass@k = 1 - C(n-c, k) / C(n, k) where n=5, c=3, k=5

┌─────────┬──────────┬──────────────────┬──────────────────┐
│ Pass@1  │ Pass@5   │ Passed Attempts  │ Total Attempts   │
│  1      │  1.0000  │      3/5         │       5          │
└─────────┴──────────┴──────────────────┴──────────────────┘
```

---

## 3. Tracing Inside the Formula

### Code Location: Lines 268-308

```python
def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    n, c, k = int(n), int(c), int(k)  ← LINE 288: Convert to integers
    
    # Validation: ensure parameters are valid
    if n <= 0 or k <= 0 or k > n:  ← LINE 291: Validate n and k
        return 0.0
    
    if c <= 0:  ← LINE 295: No correct attempts
        return 0.0
    
    if c >= n:  ← LINE 299: All attempts correct
        return 1.0
    
    if k > (n - c):  ← LINE 304: Guaranteed success case
        return 1.0
    
    # Apply the formula: 1 - C(n-c, k) / C(n, k)
    return 1.0 - comb(n - c, k) / comb(n, k)  ← LINE 308: The actual formula
```

### How to Add Debug Prints Inside Formula:

Replace line 308 with:
```python
    # Apply the formula: 1 - C(n-c, k) / C(n, k)
    numerator = comb(n - c, k)
    denominator = comb(n, k)
    result = 1.0 - numerator / denominator
    
    print(f"  Formula: 1 - C({n}-{c}, {k}) / C({n}, {k})")
    print(f"         = 1 - C({n-c}, {k}) / C({n}, {k})")
    print(f"         = 1 - {numerator} / {denominator}")
    print(f"         = 1 - {numerator/denominator:.4f}")
    print(f"         = {result:.4f}")
    
    return result
```

---

## 4. Step-by-Step Trace Example

Let's trace a specific scenario: **n=2, c=1, k=2**

### Step 1: Attempts Run (Lines 439-686)
```
Attempt 1: Model generates code → Evaluation runs → attempt1_passed = True
Attempt 2: Model generates code → Evaluation runs → attempt2_passed = False
```

### Step 2: Count Parameters (Lines 725-732)
```python
n = 2                           # Set total attempts
c = 0                           # Initialize correct count
if attempt1_passed is True:     # True → c += 1
    c += 1                      # Now c = 1
if attempt2_passed is True:     # False → skip
    c += 1
# Final: n=2, c=1
```

### Step 3: Call Formula (Line 744)
```python
pass_k_2 = estimate_pass_at_k(n=2, c=1, k=2)
```

### Step 4: Inside Formula (Lines 268-308)
```python
def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    n, c, k = int(2), int(1), int(2)  # Convert to int
    
    if n <= 0 or k <= 0 or k > n:  # 2 > 0, 2 > 0, 2 == 2 → False, continue
        return 0.0
    
    if c <= 0:  # 1 > 0 → False, continue
        return 0.0
    
    if c >= n:  # 1 < 2 → False, continue
        return 1.0
    
    if k > (n - c):  # 2 > (2 - 1) → 2 > 1 → True
        return 1.0  # Return 1.0 (guaranteed success)
```

### Step 5: Result
```
Pass@2 = 1.0
```

**Why 1.0?** Because with n=2 attempts and c=1 correct, if we pick k=2 samples, we're guaranteed to get the correct one (can't choose 2 from only 1 failure).

### Step 6: Save to Excel (Line 761)
```python
row = {
    "n_total_attempts": 2,
    "c_correct_attempts": 1,
    "k": 2,
    "pass_k_2": 1.0,
    "attempt1_passed": 1,
    "attempt2_passed": 0,
}
```

---

## 5. Breakpoint Locations for Debugging

Use these breakpoints to trace execution:

| Line | Purpose | What to Inspect |
|------|---------|-----------------|
| 725 | Start of Pass@k calculation | `task_id`, `model` |
| 729 | Count attempt 1 | `attempt1_passed`, `c` |
| 731 | Count attempt 2 | `attempt2_passed`, `c` |
| 744 | Before formula call | `n`, `c`, `k=2` |
| 288 | Inside formula (start) | `n`, `c`, `k` (parameters received) |
| 308 | Formula calculation | `n-c`, `comb(n-c,k)`, `comb(n,k)` |
| 761 | Before saving to CSV | `row` dictionary |
| 1011 | Single-run: n set | `pass_at_k_value`, `n` |
| 1012 | Single-run: c set | `passed_attempts`, `c` |
| 1023 | Single-run: formula call | `n`, `c`, `pass_at_k_value` |

---

## 6. Checking Output Files

### A. Excel File: `results/batch_results.xlsx`

Open the file and look at these columns:
1. **n_total_attempts** → Should always be 2 for batch mode
2. **c_correct_attempts** → Should be 0, 1, or 2
3. **k** → Should always be 2
4. **pass_k_2** → Should match the formula:
   - c=0 → 0.0
   - c=1 → 1.0
   - c=2 → 1.0

### B. JSON Log: `experiment_logs.jsonl`

Each attempt is logged separately:
```json
{
  "timestamp": "2026-01-29T...",
  "run_id": "batch_20260129_123456",
  "attempt_index": 1,
  "task_id": "HumanEval/7",
  "model": "codellama:7b",
  "pass_1": 0,
  "text_overlap": {
    "_debug_attempt1_passed": false,
    "humaneval_result": {
      "passed": false,
      "result": "failed: AssertionError"
    }
  }
}
```

Look for:
- `attempt_index`: 1 or 2
- `pass_1`: 0 or 1 (only for attempt 1)
- `_debug_attempt1_passed` or `_debug_attempt2_passed`: raw boolean value

---

## 7. Quick Verification Checklist

After running your code, verify:

- [ ] Excel file has columns: `n_total_attempts`, `c_correct_attempts`, `k`, `pass_k_2`
- [ ] All rows have `n_total_attempts = 2` (for batch mode with k=2)
- [ ] `c_correct_attempts` matches the sum of `attempt1_passed + attempt2_passed`
- [ ] When `c=0`, `pass_k_2 = 0.0`
- [ ] When `c=1`, `pass_k_2 = 1.0`
- [ ] When `c=2`, `pass_k_2 = 1.0`
- [ ] UI displays formula with actual n, c, k values
- [ ] Single-run shows Pass@k as a decimal (e.g., 0.9167)

---

## 8. Common Scenarios to Test

| Scenario | n | c | k | Expected Pass@k | Notes |
|----------|---|---|---|-----------------|-------|
| Both fail | 2 | 0 | 2 | 0.0 | No correct attempts |
| First passes | 2 | 1 | 2 | 1.0 | Guaranteed (k > n-c) |
| Second passes | 2 | 1 | 2 | 1.0 | Guaranteed (k > n-c) |
| Both pass | 2 | 2 | 2 | 1.0 | All correct |
| Pass@1 (half) | 2 | 1 | 1 | 0.5 | 50% chance |
| Pass@1 (none) | 2 | 0 | 1 | 0.0 | No correct |
| Pass@1 (all) | 2 | 2 | 1 | 1.0 | All correct |
| Large sample | 10 | 3 | 5 | 0.9167 | 91.67% chance |

You can test these with the `test_pass_at_k.py` script!

---

## Summary

The Pass@k formula parameters are now properly calculated and logged:
- **n** (total attempts): Set at lines 725, 1011
- **c** (correct attempts): Counted at lines 729-732, 1012
- **k** (samples): Passed to formula at lines 744, 1023
- **Pass@k** (result): Calculated at line 308, stored in Excel and displayed in UI

All parameters are now visible in the Excel output, JSON logs, and Streamlit UI! 🎉

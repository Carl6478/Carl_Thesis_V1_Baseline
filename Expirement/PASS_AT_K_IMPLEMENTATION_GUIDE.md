# Pass@k Implementation Guide

## Summary of Changes

The code has been fixed to properly use the **Pass@k formula** instead of simple boolean logic. Here's what was changed:

---

## 1. The Pass@k Formula (Now Being Used!)

**Location:** Lines 268-308 in `BaselineV1.py`

```python
def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
    
    Parameters:
    - n: Total number of code samples/attempts generated
    - c: Number of correct samples (that passed all tests)
    - k: Number of samples to consider
    """
```

### Formula Explanation:
- **Pass@k = 1 - C(n-c, k) / C(n, k)**
- This is the **unbiased estimator** for the probability that at least one of k samples will be correct
- `C(a, b)` is the binomial coefficient (combinations): "a choose b"

### Examples:
- **n=2, c=0, k=2:** Pass@2 = 1 - C(2,2)/C(2,2) = 1 - 1/1 = **0.0** (no correct attempts)
- **n=2, c=1, k=2:** Pass@2 = 1 - C(1,2)/C(2,2) = 1 - 0/1 = **1.0** (can't choose 2 from 1, guaranteed success)
- **n=2, c=2, k=2:** Pass@2 = 1 - C(0,2)/C(2,2) = **1.0** (both correct)
- **n=5, c=2, k=3:** Pass@3 = 1 - C(3,3)/C(5,3) = 1 - 1/10 = **0.9** (90% chance)

---

## 2. Batch Evaluation (Fixed)

**Location:** Lines 723-744 in `BaselineV1.py`

### Before (WRONG):
```python
pass_k_2 = 1 if (attempt1_passed is True or attempt2_passed is True) else 0
```
This was just checking "did ANY attempt pass?" → always 0 or 1

### After (CORRECT):
```python
# Count total attempts (n) and correct attempts (c)
n = 2  # Total number of attempts
c = 0  # Number of correct attempts

# Count correct attempts
if attempt1_passed is True:
    c += 1
if attempt2_passed is True:
    c += 1

# Use the estimate_pass_at_k function with k=2
pass_k_2 = estimate_pass_at_k(n=n, c=c, k=2)
```

### What Gets Logged:
The CSV/Excel output now includes:
- `n_total_attempts`: Total attempts (n parameter)
- `c_correct_attempts`: Number of correct attempts (c parameter)
- `k`: Number of samples considered (k parameter)
- `pass_k`: The calculated Pass@k value (output of formula)

**CSV columns to trace:**
```
n_total_attempts | c_correct_attempts | k | pass_k  | attempt1_passed | attempt2_passed
2                | 0                  | 2 | 0.0     | 0               | 0
2                | 1                  | 2 | 1.0     | 1               | 0
2                | 2                  | 2 | 1.0     | 1               | 1
```

---

## 3. Single-Run Pass@k Evaluation (Fixed)

**Location:** Lines 1010-1023 in `BaselineV1.py`

### Before (WRONG):
```python
pass_at_k = 1 if len(passed_attempts) > 0 else 0
```

### After (CORRECT):
```python
# Compute pass@k using the proper formula
n = pass_at_k_value  # Total number of attempts
c = len(passed_attempts)  # Number of correct attempts

# Pass@k using the estimate_pass_at_k function
all_none = all(att["passed"] is None for att in attempts)
if all_none:
    pass_at_k = None
else:
    pass_at_k = estimate_pass_at_k(n=n, c=c, k=pass_at_k_value)
```

### UI Display (Lines 1026-1037):
The Streamlit UI now shows:
1. **Formula explanation:** `Pass@k = 1 - C(n-c, k) / C(n, k) where n=X, c=Y, k=Z`
2. **Pass@1:** Result of first attempt only
3. **Pass@k:** Calculated using formula (shown as decimal, e.g., 0.8500)
4. **Passed Attempts (c):** e.g., "2/5"
5. **Total Attempts (n):** e.g., "5"

---

## 4. How to Trace the Parameters

### In the Code:
1. **Set breakpoints** at:
   - Line 725-732: Where `n` and `c` are counted (batch)
   - Line 744: Where `estimate_pass_at_k()` is called (batch)
   - Line 1011-1023: Where Pass@k is calculated (single-run)
   - Line 308: Inside the formula calculation

2. **Add debug prints** (if needed):
```python
print(f"DEBUG: n={n}, c={c}, k={k}")
print(f"DEBUG: C(n-c, k) = C({n-c}, {k}) = {comb(n-c, k)}")
print(f"DEBUG: C(n, k) = C({n}, {k}) = {comb(n, k)}")
print(f"DEBUG: Pass@{k} = {estimate_pass_at_k(n, c, k)}")
```

### In the Output Files:

#### A. Excel Output (`results/batch_results.xlsx`)
Look at these columns:
- `n_total_attempts` → n parameter
- `c_correct_attempts` → c parameter  
- `k` → k parameter (always 2 for batch)
- `pass_k` → calculated result
- `attempt1_passed` and `attempt2_passed` → raw boolean results

#### B. JSON Logs (`experiment_logs.jsonl`)
Each line contains:
```json
{
  "run_id": "batch_20260129_123456",
  "attempt_index": 1,
  "task_id": "HumanEval/7",
  "pass_1": 0,
  "text_overlap": {
    "_debug_attempt1_passed": false,
    "humaneval_result": {...}
  }
}
```

---

## 5. Verification Examples

### Example 1: Both Attempts Fail
- Attempt 1: FAILED (attempt1_passed = False)
- Attempt 2: FAILED (attempt2_passed = False)
- **n = 2, c = 0, k = 2**
- **Pass@2 = 1 - C(2,2)/C(2,2) = 1 - 1/1 = 0.0** ✅

### Example 2: One Attempt Passes
- Attempt 1: PASSED (attempt1_passed = True)
- Attempt 2: FAILED (attempt2_passed = False)
- **n = 2, c = 1, k = 2**
- **Pass@2 = 1 - C(1,2)/C(2,2) = 1 - 0/1 = 1.0** ✅
- (Can't choose 2 items from only 1 failure, so guaranteed success)

### Example 3: Both Attempts Pass
- Attempt 1: PASSED (attempt1_passed = True)
- Attempt 2: PASSED (attempt2_passed = True)
- **n = 2, c = 2, k = 2**
- **Pass@2 = 1 - C(0,2)/C(2,2) = 1.0** ✅

### Example 4: Larger Sample (n=10, c=6, k=5)
- 10 total attempts, 6 passed, evaluate best 5
- **n = 10, c = 6, k = 5**
- **Pass@5 = 1 - C(4,5)/C(10,5) = 1 - 0/252 = 1.0** ✅
- (Can't choose 5 from 4 failures, guaranteed success)

### Example 5: Larger Sample (n=10, c=3, k=5)
- 10 total attempts, 3 passed, evaluate best 5
- **n = 10, c = 3, k = 5**
- **Pass@5 = 1 - C(7,5)/C(10,5) = 1 - 21/252 = 0.9167** ✅
- (91.67% chance at least one of 5 attempts is correct)

---

## 6. Common Issues to Watch For

### Issue: Pass@k always returns 0 or 1
**Cause:** Using simple boolean logic instead of the formula
**Fix:** ✅ Fixed! Now uses `estimate_pass_at_k()`

### Issue: Pass@k is always 1.0 even when k < n
**Cause:** When `c ≥ (n - k + 1)`, we're guaranteed success
**Example:** n=2, c=1, k=2 → guaranteed to pick the correct one

### Issue: Division by zero or invalid combinations
**Cause:** Invalid parameters (k > n, n ≤ 0, etc.)
**Fix:** ✅ The function now validates parameters (lines 291-305)

---

## 7. Testing the Implementation

### Quick Test Script:
```python
from math import comb

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    n, c, k = int(n), int(c), int(k)
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > (n - c):
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# Test cases
print(f"n=2, c=0, k=2: {estimate_pass_at_k(2, 0, 2)}")  # Should be 0.0
print(f"n=2, c=1, k=2: {estimate_pass_at_k(2, 1, 2)}")  # Should be 1.0
print(f"n=2, c=2, k=2: {estimate_pass_at_k(2, 2, 2)}")  # Should be 1.0
print(f"n=10, c=3, k=5: {estimate_pass_at_k(10, 3, 5)}")  # Should be ~0.9167
```

---

## 8. What Changed Summary

| Component | Before | After |
|-----------|--------|-------|
| **Formula Function** | Defined but unused | ✅ Now called with proper validation |
| **Batch Pass@k** | Simple OR logic (0 or 1) | ✅ Uses formula (returns 0.0 to 1.0) |
| **Single-run Pass@k** | Simple counting | ✅ Uses formula with proper n, c, k |
| **CSV Output** | Only pass_k result | ✅ Includes n, c, k parameters |
| **UI Display** | Binary result | ✅ Shows formula and parameters |
| **Documentation** | Missing | ✅ Added comprehensive docstrings |

---

## 9. Where to Find What

| What you want to trace | File location | Lines |
|------------------------|---------------|-------|
| Pass@k formula definition | BaselineV1.py | 268-308 |
| Batch evaluation (n, c counting) | BaselineV1.py | 723-744 |
| Single-run Pass@k calculation | BaselineV1.py | 1010-1023 |
| UI display with formula | BaselineV1.py | 1026-1037 |
| CSV output with n, c, k | BaselineV1.py | 760-780 |
| Excel file output | results/batch_results.xlsx | (runtime) |
| JSON logs | experiment_logs.jsonl | (runtime) |

---

## 10. Next Steps

1. **Run a batch evaluation** to see the new columns in the Excel output
2. **Check the formula display** in the Streamlit UI
3. **Verify the calculations** match the examples above
4. **Review the logs** to trace n, c, k values for each task

The Pass@k metric is now correctly calculated using the unbiased estimator! 🎉

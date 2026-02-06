# Troubleshooting: All Tests Failing

If **ALL problems with ALL models** are failing Pass@k tests, there's likely a systematic issue. Here's how to diagnose and fix it.

---

## Step 1: Run the Diagnostic Script

```bash
python diagnose_failures.py
```

This will show you:
- How HumanEval tasks are structured
- What gets executed during tests
- Common failure patterns

---

## Step 2: Check What's Actually Being Executed

### In the UI:

When you run Pass@k, look at these **3 new debug views**:

1. **📝 Full Raw Model Output** (expander)
   - What the model actually generated

2. **🔍 Normalized Completion** (expander)
   - What gets added to the test after normalization

3. **🐛 Debug: Full Test Program** (NEW expander, only on failures)
   - The ACTUAL code that was executed
   - This shows prompt + completion + test combined

---

## Step 3: Common Failure Patterns

### Pattern 1: Model Generates Full Function

**What the model generates:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        ...
    return False
```

**Problem:** The prompt already has the function signature, so this creates:
```python
def has_close_elements(...):
    """docstring"""
def has_close_elements(...):  # DUPLICATE!
    for i in range(...):
```

**What normalization SHOULD do:** Remove the `def` line, keep only body

**Check:** Look at "🔍 Normalized Completion" - should NOT have `def` line

---

### Pattern 2: Model Generates Examples

**What the model generates:**
```python
has_close_elements([1.0, 2.0], 0.5)  # returns False
has_close_elements([1.0, 2.8, 3.0], 0.3)  # returns True
```

**Problem:** This isn't code - it's usage examples

**Result:** AssertionError or NameError

**Check:** Look at "📝 Full Raw Model Output" - if you see examples, the model misunderstood

---

### Pattern 3: Model Generates Doctest Format

**What the model generates:**
```python
>>> has_close_elements([1.0, 2.0], 0.5)
False
>>> has_close_elements([1.0, 2.8, 3.0], 0.3)
True
```

**Problem:** `>>>` is not valid Python syntax

**Result:** SyntaxError

**Check:** Look at "📝 Full Raw Model Output" - if you see `>>>`, the model is treating it like an interactive prompt

---

### Pattern 4: Indentation Issues

**What normalization generates:**
```python
for i in range(len(numbers)):  # Not indented!
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
```

**Problem:** When added after the function signature, it's not properly indented

**The test program becomes:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """docstring"""
for i in range(len(numbers)):  # Wrong indentation!
```

**Result:** SyntaxError or IndentationError

**Check:** Look at "🐛 Debug: Full Test Program" to see the combined code

---

## Step 4: Verify Normalization is Working

The `normalize_humaneval_completion` function should:
1. ✅ Remove the `def` line if present
2. ✅ Keep only the function body
3. ✅ Preserve indentation

### Test it manually:

```python
from BaselineV1 import normalize_humaneval_completion

# Test case: Model generated full function
completion = """def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False"""

prompt = """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    \"\"\"
"""

normalized = normalize_humaneval_completion(prompt, completion)
print("Normalized:")
print(repr(normalized))
```

**Expected output:**
```python
'    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
```

Should start with spaces (indented), NOT with `def`.

---

## Step 5: Check Model Behavior

### Which model are you using?

```bash
# Check installed models
ollama list
```

**General models (bad for code):**
- `mistral:7b` - General purpose, often generates examples
- `llama2:7b` - General purpose, poor at code
- `qwen:7b` - General purpose

**Code-specific models (better):**
- `codellama:7b` ✅
- `deepseek-coder:6.7b` ✅
- `qwen2.5-coder:7b` ✅
- `starcoder2:7b` ✅

### Try a code model:

```bash
ollama pull codellama:7b
```

Then test with it.

---

## Step 6: Look at experiment_logs.jsonl

After running tests, check the logs:

```python
import json

with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Get the most recent failure
recent = logs[-1]

print("Model:", recent['model'])
print("Task:", recent['task_id'])
print("\nRaw Output:")
print(recent['raw_output'])
print("\nExtracted Code:")
print(recent['extracted_code'])

# Check if debug_program is in the result
if 'text_overlap' in recent and 'humaneval_result' in recent['text_overlap']:
    result = recent['text_overlap']['humaneval_result']
    if 'debug_program' in result:
        print("\nDebug Program (what was executed):")
        print(result['debug_program'])
```

---

## Step 7: Manual Test

Try testing with known good code:

```python
from human_eval.data import read_problems

problems = read_problems()
task = problems["HumanEval/0"]

# Use the canonical solution
completion = task["canonical_solution"]

print("Canonical solution:")
print(completion)

# Build test program
check_program = (
    task["prompt"] + completion + "\n" +
    task["test"] + "\n" +
    f"check({task['entry_point']})"
)

print("\nTest program:")
print(check_program)

# Execute
exec_globals = {}
try:
    exec(check_program, exec_globals)
    print("\n✅ PASSED!")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
```

**If this fails:** The test execution itself is broken  
**If this passes:** The models are generating bad code

---

## Step 8: Check Specific Error Messages

### "SyntaxError: invalid syntax"
→ Model generated non-Python code (examples, doctest, etc.)  
→ Check "📝 Full Raw Model Output"

### "AssertionError: assertion failed"
→ Code runs but produces wrong output  
→ Model's logic is incorrect  
→ Check "🔍 Normalized Completion" - is it actual code?

### "NameError: name 'X' is not defined"
→ Model is calling undefined functions  
→ Check "🐛 Debug: Full Test Program"

### "IndentationError"
→ Normalization broke indentation  
→ Check "🔍 Normalized Completion" - should start with spaces

### "failed: empty completion"
→ Model didn't generate anything  
→ Check "📝 Full Raw Model Output" - is it empty?

---

## Step 9: Potential Code Issues to Check

### Issue 1: extract_python_code is too aggressive

Check if `extract_python_code` is removing too much:

```python
# In BaselineV1.py, around line 138
def extract_python_code(output: str) -> str:
    # Check what this is doing
```

If this is removing actual code, extracted_code will be empty.

### Issue 2: normalize_humaneval_completion is wrong

Check if normalization is breaking code:

```python
# In BaselineV1.py, around line 284
def normalize_humaneval_completion(prompt: str, completion: str) -> str:
    # Check what this returns
```

Should preserve indentation and function body.

### Issue 3: Test assembly is wrong

Check line ~197:
```python
check_program = (
    task["prompt"] + completion + "\n" +
    task["test"] + "\n" +
    f"check({task['entry_point']})"
)
```

This should create valid Python when completion is the function body.

---

## Step 10: Expected Success Pattern

When working correctly, you should see:

**📝 Full Raw Model Output:**
```
Here's the solution:

def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**✂️ Extracted Code:**
```python
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**🔍 Normalized Completion:**
```python
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**HumanEval result:**
```json
{
  "passed": true,
  "result": "passed",
  "task_id": "HumanEval/0"
}
```

---

## Quick Checklist

- [ ] Run `diagnose_failures.py`
- [ ] Check "🐛 Debug: Full Test Program" expander in UI
- [ ] Verify model is generating code (not examples)
- [ ] Try a code-specific model (codellama)
- [ ] Check normalization preserves indentation
- [ ] Test with canonical solution manually
- [ ] Look at `experiment_logs.jsonl` for patterns
- [ ] Verify extract_python_code isn't too aggressive

---

## Contact Point

If still failing after all checks:
1. Run diagnosis script
2. Capture the "🐛 Debug: Full Test Program" output
3. Share the actual test program that's being executed
4. Share the raw model output

This will show exactly where the pipeline is breaking! 🔍

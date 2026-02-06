# How to See Model Output

There are multiple ways to see what the model generated, depending on what information you need.

---

## 1. In the Streamlit UI (Live View)

### During Pass@k Evaluation

For each attempt, you'll see **3 types of output**:

#### A. 📝 Full Raw Model Output (Expander)
Click to expand and see **exactly what the model generated** before any processing.

```
Example:
Sure! Here's the implementation:

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This function checks if any two elements are closer than the threshold.
```

**Use this to:**
- See if model is generating code at all
- Check model's explanation/reasoning
- Debug extraction issues
- Understand model behavior

---

#### B. ✂️ Extracted Code
Shows the code **after extraction** (markdown removed, code isolated).

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Use this to:**
- Verify extraction worked correctly
- See what code will be analyzed
- Check if syntax looks valid

---

#### C. 🔍 Normalized Completion (Expander)
Shows **only the function body** that gets tested (the actual completion without the def line).

```python
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Use this to:**
- See exactly what code is executed in tests
- Debug test failures
- Understand why syntax errors occur

---

### During Single Run (No Pass@k)

You'll see:

#### 🧠 Full Model Output
The complete raw output from the model.

#### 🧩 Extracted Code Used for Analysis
The code after extraction that will be analyzed for metrics.

---

## 2. In Log Files (Permanent Record)

All model outputs are saved to log files for later analysis.

### A. JSON Log File: `experiment_logs.jsonl`

**Location:** Same directory as your script  
**Format:** JSON Lines (one JSON object per line)

**Each line contains:**
```json
{
  "timestamp": "2026-01-29T15:30:45.123456",
  "run_id": "batch_20260129_153045",
  "attempt_index": 1,
  "model": "codellama:7b",
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(...):",
  "rag_enabled": false,
  "raw_output": "Sure! Here's the implementation:\n\n```python\ndef has_close_elements(...):",
  "extracted_code": "def has_close_elements(...):\n    for i in range...",
  "metrics": {"lines_of_code": 15, "cyclomatic_complexity": 5},
  "text_overlap": {
    "humaneval_result": {"passed": true, "result": "passed"},
    "_debug_attempt1_passed": true
  },
  "similarity_score": null,
  "pass_1": 1
}
```

**How to read it:**
```bash
# View last 10 entries
tail -n 10 experiment_logs.jsonl

# Search for specific model
grep "codellama:7b" experiment_logs.jsonl

# Pretty print a specific line (use Python or jq)
cat experiment_logs.jsonl | tail -n 1 | python -m json.tool
```

**Use this to:**
- Review all attempts after batch runs
- Extract raw outputs for manual analysis
- Build custom analyses/visualizations
- Debug what happened in past runs

---

### B. Excel File: `results/batch_results.xlsx`

**Location:** `results/batch_results.xlsx`  
**Format:** Excel workbook with one sheet per run

**Contains aggregated data:**
```
model | task_id | n | c | pass_k | attempt1_passed | attempt1_result | attempt1_CC | ...
codellama:7b | HumanEval/0 | 2 | 1 | 1.0 | 1 | passed | 5.0 | ...
```

**Columns include:**
- `attempt1_result`, `attempt2_result` - Test result strings
- `attempt1_passed_raw`, `attempt2_passed_raw` - Boolean pass/fail
- All metrics (CC, LOC, cognitive, halstead)

**Note:** Raw output is **not** in Excel (too large). Use JSON logs for that.

**Use this to:**
- Statistical analysis in Excel/Python
- Create charts and tables
- Calculate aggregate metrics
- Compare model performance

---

## 3. Programmatic Access

### Reading JSON Logs with Python

```python
import json

# Read all logs
with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Find all outputs from a specific model
codellama_outputs = [
    log for log in logs 
    if log['model'] == 'codellama:7b'
]

# Get raw outputs that failed
failed_outputs = [
    log['raw_output'] 
    for log in logs 
    if log['pass_1'] == 0
]

# Print a specific output
for log in logs:
    if log['task_id'] == 'HumanEval/0':
        print(f"Model: {log['model']}")
        print(f"Output: {log['raw_output']}")
        print(f"Passed: {log['pass_1']}")
        print("---")
```

### Reading Excel with Pandas

```python
import pandas as pd

# Read the latest sheet
df = pd.read_excel('results/batch_results.xlsx', sheet_name='batch_20260129_153045')

# See all attempt results
print(df[['model', 'task_id', 'attempt1_result', 'attempt2_result']])

# Filter failed attempts
failed = df[df['pass_k'] == 0.0]
print(failed[['model', 'task_id', 'attempt1_result']])
```

---

## 4. Quick Reference: Where to Find What

| What You Want | Where to Look | When to Use |
|---------------|---------------|-------------|
| **Full raw output** | UI: 📝 expander | During run (debugging) |
| **Extracted code** | UI: ✂️ section | During run (verify extraction) |
| **Tested code** | UI: 🔍 expander | During run (debug failures) |
| **Historical outputs** | `experiment_logs.jsonl` | After run (analysis) |
| **Aggregate stats** | `results/batch_results.xlsx` | After run (statistics) |
| **Error messages** | UI: HumanEval result JSON | During run (debug failures) |
| **All attempts for one run** | JSON log (filter by run_id) | After batch (review) |

---

## 5. Common Use Cases

### Use Case 1: "Why did this attempt fail?"

**During the run:**
1. Click **🔍 Normalized Completion** - See what code was tested
2. Look at **HumanEval result JSON** - See the error message
3. Click **📝 Full Raw Model Output** - See if model misunderstood

**Example:**
```
Normalized Completion:
    has_close_elements([1.0, 2.0], 0.5)  # ← This is an example, not code!
    
HumanEval Result:
    "result": "failed: AssertionError: assertion failed"
    
Conclusion: Model generated examples instead of implementation
```

---

### Use Case 2: "What did the model generate for all attempts?"

**After the run:**
```python
import json

# Load logs
with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Get specific task
task_logs = [l for l in logs if l['task_id'] == 'HumanEval/0']

# Print all outputs
for i, log in enumerate(task_logs, 1):
    print(f"\n=== Attempt {i} ===")
    print(f"Model: {log['model']}")
    print(f"Passed: {log['pass_1']}")
    print(f"Output:\n{log['raw_output']}")
```

---

### Use Case 3: "Compare outputs from different models"

```python
import json
import pandas as pd

# Load logs
with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Create comparison dataframe
df = pd.DataFrame([{
    'model': log['model'],
    'task': log['task_id'],
    'passed': log['pass_1'],
    'output_length': len(log['raw_output']),
    'has_code': len(log['extracted_code']) > 0
} for log in logs])

# Compare models
print(df.groupby('model').agg({
    'passed': 'mean',
    'output_length': 'mean',
    'has_code': 'mean'
}))
```

---

### Use Case 4: "Show me all failed attempts with their outputs"

```python
import json

with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

failed = [l for l in logs if l['pass_1'] == 0]

for log in failed:
    print(f"\nTask: {log['task_id']}")
    print(f"Model: {log['model']}")
    print(f"Error: {log['text_overlap'].get('humaneval_result', {}).get('result', 'unknown')}")
    print(f"Generated:\n{log['raw_output'][:200]}...")  # First 200 chars
```

---

## 6. Viewing During Batch Evaluation

During batch runs, outputs are **logged but not displayed** (too many to show in UI).

**To see them:**
1. **Wait for batch to complete**
2. **Open `experiment_logs.jsonl`**
3. **Filter by run_id** (shown at the end)

```python
import json

run_id = "batch_20260129_153045"  # From batch completion message

with open('experiment_logs.jsonl', 'r') as f:
    batch_logs = [json.loads(line) for line in f if json.loads(line)['run_id'] == run_id]

# Now you have all outputs from that batch
for log in batch_logs:
    print(f"{log['model']} × {log['task_id']}: {log['pass_1']}")
```

---

## 7. Example Output Viewing Session

```python
# After running a batch evaluation
import json
import pandas as pd

# 1. Load all logs
with open('experiment_logs.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# 2. Filter to your recent run
run_id = "batch_20260129_153045"
run_logs = [l for l in logs if l.get('run_id') == run_id]

print(f"Found {len(run_logs)} attempts in this run\n")

# 3. See summary
df = pd.DataFrame([{
    'model': l['model'],
    'task': l['task_id'],
    'attempt': l.get('attempt_index', 0),
    'passed': l['pass_1']
} for l in run_logs])

print(df.groupby(['model', 'task'])['passed'].sum())

# 4. Look at specific failure
failed_log = next(l for l in run_logs if l['pass_1'] == 0)
print(f"\nExample failure:")
print(f"Task: {failed_log['task_id']}")
print(f"Model: {failed_log['model']}")
print(f"\nModel output:")
print(failed_log['raw_output'])
```

---

## Summary

✅ **Live viewing:** Use Streamlit UI expanders (📝, 🔍, ✂️)  
✅ **Permanent record:** `experiment_logs.jsonl` has everything  
✅ **Statistics:** `results/batch_results.xlsx` for aggregate analysis  
✅ **Programmatic:** Python scripts for custom analysis  

All model outputs are captured and accessible! 🎯

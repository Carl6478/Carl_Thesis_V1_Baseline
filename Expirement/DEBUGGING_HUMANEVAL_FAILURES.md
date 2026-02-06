# Debugging HumanEval Test Failures

## Understanding Test Results

When you run Pass@k evaluation, you may see different types of failures. Here's how to understand and fix them:

---

## Common Failure Types

### 1. Empty Completion
```json
{
  "passed": false,
  "result": "failed: empty completion"
}
```

**Cause:** The model didn't generate any code, or code extraction failed.

**How to debug:**
- Check the "Raw model output" section in the UI
- Look for the extracted code - is it empty?
- The model might have generated text instead of code

**Fix:**
- Try a different model (some models are better at code generation)
- Adjust your prompt to be clearer about code generation
- Check if the model is properly installed and running

---

### 2. Syntax Error
```json
{
  "passed": false,
  "result": "failed: SyntaxError: unterminated string literal (detected at line 12)",
  "completion": "...",
  "error_line": 12
}
```

**Cause:** The generated code has invalid Python syntax.

**How to debug:**
- Click "🔍 Normalized Completion" expander to see the code being tested
- Look at the error line number
- Common issues:
  - Unterminated strings (missing quotes)
  - Mismatched parentheses/brackets
  - Invalid indentation
  - Incomplete code

**Why this happens:**
- The model generated incomplete code
- Code extraction cut off in the middle of a statement
- The model doesn't understand Python syntax well

**Fix:**
- Use a better code-generation model (e.g., codellama, deepseek-coder)
- Check if code extraction is working correctly
- The model might need better prompting

---

### 3. Assertion Error
```json
{
  "passed": false,
  "result": "failed: AssertionError: Expected 5, got 3",
  "completion": "..."
}
```

**Cause:** The code runs but produces wrong output.

**How to debug:**
- Click "🔍 Normalized Completion" to see the generated function
- The code is syntactically correct but logically wrong
- Check the test cases in the task to understand what's expected

**Why this happens:**
- The model misunderstood the problem
- Logic error in generated code
- Edge cases not handled

**This is normal!** Not all model attempts will be correct. That's why we use Pass@k - to measure how often the model gets it right.

---

### 4. Runtime Error (NameError, TypeError, etc.)
```json
{
  "passed": false,
  "result": "failed: NameError: name 'helper_func' is not defined",
  "completion": "..."
}
```

**Cause:** The code tries to use something that doesn't exist.

**Common errors:**
- `NameError`: Using undefined variables/functions
- `TypeError`: Wrong types (e.g., adding string + int)
- `AttributeError`: Calling methods that don't exist
- `IndexError`: Array index out of bounds

**How to debug:**
- Look at the normalized completion
- Check if the model is trying to call functions that weren't defined
- See if variable names are misspelled

---

### 5. Timeout
```json
{
  "passed": false,
  "result": "timeout"
}
```

**Cause:** Code took too long to execute (> 30 seconds).

**Why this happens:**
- Infinite loop in generated code
- Very inefficient algorithm
- Recursive function without base case

**How to debug:**
- The code is likely stuck in a loop
- Check the normalized completion for obvious infinite loops

---

## Diagnostic Features in the UI

### 1. Raw Model Output
Shows exactly what the model generated before code extraction.

**Use this to:**
- See if the model is generating code at all
- Check if code extraction is working
- Understand the model's response format

### 2. Extracted Code
The code after extraction (removes markdown, text, etc.).

**Use this to:**
- Verify code extraction worked correctly
- See what code will be analyzed for metrics

### 3. Normalized Completion (in expander)
The actual code that gets tested (function body only, not the def line).

**Use this to:**
- See exactly what code is being executed
- Debug syntax errors and logic errors
- Understand why tests fail

### 4. HumanEval Result JSON
The detailed test result with error messages.

**Use this to:**
- See pass/fail status
- Read error messages
- Check error line numbers (for syntax errors)

---

## Example: Debugging a Syntax Error

Let's say you see:
```
Attempt 1 ❌ FAILED
HumanEval result:
{
  "passed": false,
  "result": "failed: SyntaxError: unterminated string literal (detected at line 12)"
}
```

**Step-by-step debugging:**

1. **Click the "🔍 Normalized Completion" expander**
   ```python
   # You might see something like:
   result = []
   for item in items:
       if item.startswith("hello):  # ← Missing closing quote!
           result.append(item)
   return result
   ```

2. **Identify the problem**
   - Line 12 has `"hello)` instead of `"hello")`
   - The model made a typo

3. **Understand why**
   - This is a generation error - the model isn't perfect
   - This is why we use Pass@k - try multiple times and pick the best

4. **Check other attempts**
   - Attempt 2 might have generated correct code
   - That's the whole point of Pass@k!

---

## Understanding Pass@k Results

### Example Output:
```
Pass@k Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formula: Pass@k = 1 - C(n-c, k) / C(n, k) where n=5, c=2, k=5

Pass@1: 0             ← First attempt failed
Pass@5: 1.0000        ← But at least one of 5 attempts passed!
Passed Attempts (c): 2/5
Total Attempts (n): 5
```

**What this means:**
- n=5: Generated 5 code samples
- c=2: 2 of them passed tests
- k=5: Evaluated all 5
- Pass@5 = 1.0: 100% chance (guaranteed to have at least one correct)
- Pass@1 = 0: First attempt failed (but others succeeded!)

**This is success!** Even though the first attempt failed, the model eventually generated correct code.

---

## Common Issues and Solutions

### Issue: All attempts fail with empty completion
**Solution:**
- Model might not be responding
- Check if Ollama is running: `ollama list`
- Try a different model
- Simplify the prompt

### Issue: All attempts fail with syntax errors
**Solution:**
- Model isn't good at Python
- Try a code-specific model (codellama, deepseek-coder)
- Check if code extraction is cutting off code mid-statement

### Issue: All attempts fail with assertion errors
**Solution:**
- Model doesn't understand the task
- This is normal for difficult tasks
- Try more attempts (increase k)
- Results help evaluate model quality

### Issue: Mix of failures and passes
**Solution:**
- **This is expected!** Models aren't perfect
- This is why Pass@k exists
- Your metrics show model reliability
- c/n ratio shows success rate

---

## Interpreting Results for Research

### Good Model Performance:
```
Task: HumanEval/0
n=5, c=4, k=5
Pass@1: 1 (first try worked!)
Pass@5: 1.0 (guaranteed success)
```

### Moderate Performance:
```
Task: HumanEval/0
n=5, c=2, k=5
Pass@1: 0 (first failed)
Pass@5: 0.9 (90% chance with 5 tries)
```

### Poor Performance:
```
Task: HumanEval/0
n=5, c=0, k=5
Pass@1: 0
Pass@5: 0.0 (no correct solutions)
```

### Your Research Question:
"Does the model improve with RAG/prompting?" 

Compare:
- **Baseline**: Pass@k without RAG
- **With RAG**: Pass@k with reference docs
- **Expected**: Higher c (more correct) with RAG → Higher Pass@k

---

## Quick Reference: Error Types

| Error | Meaning | Model Issue | Research Impact |
|-------|---------|-------------|-----------------|
| Empty completion | No code generated | Can't solve | Model failed completely |
| Syntax error | Invalid Python | Can't write syntax | Model lacks basic Python |
| Assertion error | Wrong output | Wrong logic | Model misunderstood task |
| Runtime error | Crashes | Bad logic | Model made mistakes |
| Timeout | Too slow/infinite loop | Inefficient | Algorithm issue |
| Passed | Correct! | Solved correctly | Success! |

---

## Next Steps

1. **Run your evaluation** - failures are expected and valuable data
2. **Review the normalized completions** - understand what code was generated  
3. **Calculate Pass@k** - aggregate across all attempts
4. **Compare conditions** - RAG vs baseline, different models, etc.
5. **Analyze patterns** - which tasks are harder? which models do better?

Remember: **Failures are data!** They help you understand model capabilities and limitations. That's the whole point of the research! 🎯

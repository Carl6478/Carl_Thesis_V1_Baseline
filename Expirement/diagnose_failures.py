"""
Diagnostic script to understand why all Pass@k tests are failing.
Run this to analyze the test execution process.
"""

import sys

# Check if human_eval is available
try:
    from human_eval.data import read_problems
    print("✅ human_eval.data available")
    problems = read_problems()
    print(f"✅ Loaded {len(problems)} HumanEval problems")
except Exception as e:
    print(f"❌ human_eval.data error: {e}")
    sys.exit(1)

# Get a sample problem
task_id = "HumanEval/0"
task = problems[task_id]

print("\n" + "="*80)
print(f"TASK: {task_id}")
print("="*80)

print("\n1. PROMPT (Function signature + docstring):")
print("-" * 80)
print(task["prompt"])
print("-" * 80)

print("\n2. ENTRY POINT:")
print(f"   {task['entry_point']}")

print("\n3. TEST CODE:")
print("-" * 80)
print(task["test"])
print("-" * 80)

print("\n4. CANONICAL SOLUTION (for reference):")
print("-" * 80)
print(task["canonical_solution"])
print("-" * 80)

# Simulate what the test execution does
print("\n" + "="*80)
print("SIMULATING TEST EXECUTION")
print("="*80)

# Example completion (what a model might generate)
example_completions = [
    # Case 1: Model generates full function
    '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False''',
    
    # Case 2: Model generates just body (indented)
    '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False''',
    
    # Case 3: Model generates just body (not indented)
    '''for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
        if idx != idx2:
            distance = abs(elem - elem2)
            if distance < threshold:
                return True
return False''',
]

for i, completion in enumerate(example_completions, 1):
    print(f"\n--- CASE {i}: Completion Type ---")
    print("Completion:")
    print(repr(completion[:100]))
    
    # Build the test program as the code does
    check_program = (
        task["prompt"] + completion + "\n" +
        task["test"] + "\n" +
        f"check({task['entry_point']})"
    )
    
    print(f"\nFull test program (first 500 chars):")
    print("-" * 80)
    print(check_program[:500])
    print("-" * 80)
    
    # Try to execute
    exec_globals = {}
    try:
        exec(check_program, exec_globals)
        print("✅ PASSED!")
    except SyntaxError as e:
        print(f"❌ SyntaxError: {e}")
        print(f"   Line {e.lineno}: {e.text}")
    except AssertionError as e:
        print(f"❌ AssertionError: {e}")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
The test execution combines:
1. task["prompt"] - Function signature + docstring
2. completion - What the model generates
3. task["test"] - Test function definition
4. check(entry_point) - Call to test function

Common failures:
- If model generates FULL function → duplicate definition
- If model generates UNINDENTED body → indentation error  
- If model generates EXAMPLES → not executable code
- If model generates WRONG logic → assertion error

Check the 'normalized completion' in the UI to see what's actually being used.
""")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
1. Run a Pass@k test and look at:
   - 📝 Full Raw Model Output
   - 🔍 Normalized Completion  
   - 🐛 Debug: Full Test Program (new expander!)
   
2. Check if models are generating:
   - Full functions (def ...) → should be normalized to body only
   - Examples (function calls) → will fail
   - Doctest format (>>>) → will fail
   
3. Try a code-specific model:
   ollama pull codellama:7b
   
4. Check experiment_logs.jsonl for "debug_program" field to see what's being executed
""")

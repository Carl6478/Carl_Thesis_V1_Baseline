"""
Test script to verify normalization preserves indentation correctly
"""

# Simulate the fixed normalization function
def normalize_humaneval_completion_fixed(prompt: str, completion: str) -> str:
    """Fixed version with proper indentation handling"""
    completion = completion.strip()
    if not completion:
        return completion
   
    lines = completion.splitlines()
    if lines and lines[0].strip().startswith("def "):
        body_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith("def "):
                in_function = True
                continue
            if in_function:
                stripped = line.strip()
                if stripped and not line.startswith((" ", "\t")) and (
                    stripped.startswith("def ") or
                    stripped.startswith("class ") or
                    stripped.startswith("@")
                ):
                    break
                body_lines.append(line)
        
        if body_lines:
            # Remove trailing empty lines only
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            
            # Ensure minimum indentation is at least 4 spaces
            min_indent = float('inf')
            for line in body_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # If minimum indentation is 0, add 4 spaces to all non-empty lines
            if min_indent == 0 or min_indent == float('inf'):
                body_lines = ["    " + line if line.strip() else line for line in body_lines]
            
            completion = "\n".join(body_lines)
        else:
            completion = ""
   
    return completion


# Test with CodeLlama's actual output
codellama_output = """def has_close_elements(numbers, threshold):
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[j] - numbers[i]) < threshold:
                return True
    return False"""

print("="*80)
print("INPUT (CodeLlama's extracted code):")
print("="*80)
print(codellama_output)
print()

print("="*80)
print("OUTPUT (Normalized completion):")
print("="*80)
normalized = normalize_humaneval_completion_fixed("", codellama_output)
print(normalized)
print()

print("="*80)
print("VERIFICATION:")
print("="*80)
lines = normalized.split('\n')
for i, line in enumerate(lines, 1):
    if line.strip():
        indent = len(line) - len(line.lstrip())
        print(f"Line {i}: {indent} spaces - {repr(line[:50])}")

print()
print("="*80)
print("TEST: Build full program")
print("="*80)

prompt = '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

test_code = '''from typing import List

''' + prompt + normalized + '''

def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True

check(has_close_elements)'''

print(test_code)
print()

print("="*80)
print("EXECUTING TEST:")
print("="*80)

try:
    exec_globals = {}
    exec(test_code, exec_globals)
    print("✅ SUCCESS! Code executed without errors!")
except SyntaxError as e:
    print(f"❌ SyntaxError: {e}")
    print(f"   Line {e.lineno}: {e.text}")
except AssertionError as e:
    print(f"❌ AssertionError: {e}")
    print("   (Logic is wrong but syntax is correct)")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")

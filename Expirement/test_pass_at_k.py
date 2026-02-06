"""
Test script to verify Pass@k formula implementation
Run this to validate the formula works correctly before using it in the main app
"""

from math import comb


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate Pass@k using the unbiased estimator formula.
    
    Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
    
    Parameters:
    - n: Total number of code samples/attempts generated
    - c: Number of correct samples (that passed all tests)
    - k: Number of samples to consider
    
    Returns:
    - Float between 0.0 and 1.0
    """
    n, c, k = int(n), int(c), int(k)
    
    # Validation
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > (n - c):
        return 1.0
    
    # Apply the formula
    return 1.0 - comb(n - c, k) / comb(n, k)


def test_pass_at_k():
    """Run test cases to verify the formula"""
    
    test_cases = [
        # (n, c, k, expected_result, description)
        (2, 0, 2, 0.0, "Both attempts fail"),
        (2, 1, 2, 1.0, "One of two attempts passes"),
        (2, 2, 2, 1.0, "Both attempts pass"),
        (2, 1, 1, 0.5, "Pass@1 with 1/2 correct"),
        (2, 2, 1, 1.0, "Pass@1 with 2/2 correct"),
        (10, 3, 5, 0.9167, "10 attempts, 3 correct, pick 5"),
        (10, 6, 5, 1.0, "10 attempts, 6 correct, pick 5 (guaranteed)"),
        (10, 1, 1, 0.1, "Pass@1 with 1/10 correct"),
        (5, 2, 3, 0.9, "5 attempts, 2 correct, pick 3"),
        (100, 50, 10, 0.9994, "Large sample, half correct"),
    ]
    
    print("=" * 80)
    print("Pass@k Formula Test Results")
    print("=" * 80)
    print(f"{'n':>4} {'c':>4} {'k':>4} | {'Expected':>10} {'Actual':>10} {'Match':>8} | Description")
    print("-" * 80)
    
    all_passed = True
    
    for n, c, k, expected, description in test_cases:
        actual = estimate_pass_at_k(n, c, k)
        match = abs(actual - expected) < 0.01  # Allow small floating point differences
        status = "PASS" if match else "FAIL"
        
        if not match:
            all_passed = False
        
        print(f"{n:4} {c:4} {k:4} | {expected:10.4f} {actual:10.4f} {status:>8} | {description}")
    
    print("-" * 80)
    
    if all_passed:
        print("[SUCCESS] All tests passed! The Pass@k formula is working correctly.")
    else:
        print("[FAILURE] Some tests failed. Please review the implementation.")
    
    print("=" * 80)
    
    # Additional manual calculation examples
    print("\n" + "=" * 80)
    print("Manual Calculation Examples")
    print("=" * 80)
    
    examples = [
        (2, 0, 2),
        (2, 1, 2),
        (2, 2, 2),
        (10, 3, 5),
    ]
    
    for n, c, k in examples:
        result = estimate_pass_at_k(n, c, k)
        
        # Show the calculation steps
        print(f"\nExample: n={n}, c={c}, k={k}")
        print(f"  Formula: Pass@{k} = 1 - C(n-c, k) / C(n, k)")
        print(f"         = 1 - C({n}-{c}, {k}) / C({n}, {k})")
        print(f"         = 1 - C({n-c}, {k}) / C({n}, {k})")
        
        if k > (n - c):
            print(f"         = 1.0 (can't choose {k} from {n-c} failures)")
        elif c <= 0:
            print(f"         = 0.0 (no correct attempts)")
        elif c >= n:
            print(f"         = 1.0 (all attempts correct)")
        else:
            numerator = comb(n - c, k)
            denominator = comb(n, k)
            print(f"         = 1 - {numerator} / {denominator}")
            print(f"         = 1 - {numerator / denominator:.4f}")
        
        print(f"  Result: Pass@{k} = {result:.4f}")
    
    print("=" * 80)


if __name__ == "__main__":
    test_pass_at_k()

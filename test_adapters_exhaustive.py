from adapters import (
    AdapterRegistry,
    ArithmeticInputAdapter,
    AlgebraInputAdapter,
    CalculusInputAdapter,
)

registry = AdapterRegistry([
    ArithmeticInputAdapter(),
    AlgebraInputAdapter(),
    CalculusInputAdapter(),
])

# --------------------------------------------------
# Test cases
# --------------------------------------------------

TEST_CASES = {
    # Arithmetic
    "6+5": "add",
    "432+245": "add",
    "0003+004": "add",

    # Algebra
    "5x + 7y = 0": "solve_linear",
    "(x + x)^2": "simplify",
    "x^2 + 3x + 2": "simplify",

    # Calculus
    "d/dx(x^2)": "differentiate",
    "d/dx(x^3 + x)": "differentiate",

    # Ambiguous (priority test)
    "d/dx(x + x)": "differentiate",   # calculus must win
    "x + x": "simplify",               # algebra wins
}

UNSUPPORTED_CASES = [
    "hello world",
    "2**3",
    "sin(x)",
    "",
]


def run_supported_tests():
    print("\n=== SUPPORTED INPUT TESTS ===\n")

    for inp, expected_goal in TEST_CASES.items():
        state = registry.adapt(inp)
        actual_goal = state.control.goal

        status = "PASS" if actual_goal == expected_goal else "FAIL"

        print(f"INPUT: {inp}")
        print(f"EXPECTED: {expected_goal}")
        print(f"ACTUAL:   {actual_goal}")
        print(f"RESULT:   {status}")
        print("-" * 40)



def run_unsupported_tests():
    print("\n=== UNSUPPORTED INPUT TESTS ===\n")

    for inp in UNSUPPORTED_CASES:
        try:
            registry.adapt(inp)
            print(f"INPUT: {inp}")
            print("RESULT: FAIL (should have raised error)")
        except ValueError as e:
            print(f"INPUT: {inp}")
            print("RESULT: PASS (error raised)")
        print("-" * 40)


if __name__ == "__main__":
    run_supported_tests()
    run_unsupported_tests()

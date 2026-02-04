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

tests = [
    "6+5",
    "432+245",
    "5x + 7y = 0",
    "d/dx(x^2)",
]

for t in tests:
    state = registry.adapt(t)
    print(t)
    print(state["control"]["goal"])
    print("-" * 30)

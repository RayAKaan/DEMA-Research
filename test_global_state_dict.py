from state import GlobalState

state = GlobalState()

state.control.goal = "solve_linear"
state.algebra.expr = "5x + 7y = 0"

d = state.to_dict()

print(type(d))
print(d["control"]["goal"])
print(d["algebra"]["expr"])

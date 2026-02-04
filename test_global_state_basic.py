from state import GlobalState

state = GlobalState()

print("GlobalState created")

state.control.goal = "add"
state.arithmetic.digits_a = [1, 2]
state.arithmetic.digits_b = [3, 4]

print("Goal:", state.control.goal)
print("Digits A:", state.arithmetic.digits_a)
print("Digits B:", state.arithmetic.digits_b)

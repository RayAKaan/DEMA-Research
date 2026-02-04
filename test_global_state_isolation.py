from state import GlobalState

s1 = GlobalState()
s2 = GlobalState()

s1.arithmetic.digits_a = [9]
s2.arithmetic.digits_a = [1]

print("S1 digits:", s1.arithmetic.digits_a)
print("S2 digits:", s2.arithmetic.digits_a)

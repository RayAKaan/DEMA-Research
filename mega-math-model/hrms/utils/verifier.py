from sympy import sympify, simplify
from sympy.core.sympify import SympifyError

def verify_equivalence(pred: str, gold: str) -> bool:
    try:
        pp = sympify(pred)
        gg = sympify(gold)
        diff = simplify(pp - gg)
        return diff == 0
    except SympifyError:
        return pred.strip() == gold.strip()
    except:
        return False

from sympy import sympify
from sympy.core.sympify import SympifyError

def normalize_expr(expr_str: str):
    try:
        return str(sympify(expr_str))
    except SympifyError:
        return " ".join(expr_str.strip().split())

def safe_normalize(expr_str: str):
    try:
        return normalize_expr(expr_str)
    except:
        return expr_str

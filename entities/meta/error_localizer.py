from hrms.base import HRMBase

class ErrorLocalizerHRM(HRMBase):
    def __init__(self, name='Error_Localizer'):
        super().__init__(name=name)

    def forward(self, tokens):
        return [['error_at_term_1']], {'conf': 0.5}

def make_dataset(n=5000):
    examples = []
    for _ in range(n):
        inp = '3*x + 2*x'
        attempt = '6*x'
        gold = '5*x'
        examples.append(((inp, attempt, gold), 'term_1'))
    return examples

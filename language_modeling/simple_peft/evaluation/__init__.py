from .commonsense_170k import Commonsense170kEvaluator
from .math import OpenAIGSM8kEvaluator, GSM8kEvaluator, SVAMPEvaluator, AQuAEvaluator, MAWPSEvaluator

EVALUATORS = {
    "commonsense_170k": Commonsense170kEvaluator,
    "openai/gsm8k": OpenAIGSM8kEvaluator,
    "gsm8k": GSM8kEvaluator,
    "SVAMP": SVAMPEvaluator,
    "AQuA": AQuAEvaluator,
    "mawps": MAWPSEvaluator,
}
commonsense_170k_eval_sets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
for eval_set in commonsense_170k_eval_sets:
    EVALUATORS[eval_set] = Commonsense170kEvaluator

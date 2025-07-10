import re
from .evaluator_base import CausalLMEvaluator, extract_output


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def get_float(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """ 

    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance. 

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer
    

class OpenAIGSM8kEvaluator(CausalLMEvaluator):

    def compare(self, example, raw_generation):
        # check if generation is correct
        answer = extract_output(example[self.response_field], self.label_trigger_tokens)
        generation = extract_answer_number(raw_generation)
        is_correct = abs(float(extract_answer_number(answer)) - generation) <= 0.001
        return is_correct, {"generation": generation, "answer": answer, "is_correct": is_correct}
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/gsm8k": correct_count / total_count}


class GSM8kEvaluator(CausalLMEvaluator):

    def compare(self, example, raw_generation):
        # check if generation is correct
        answer = example["answer"]
        generation = extract_answer_number(raw_generation)
        is_correct = abs(float(extract_answer_number(answer)) - generation) <= 0.001
        return is_correct, {"generation": generation, "answer": answer, "is_correct": is_correct}
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/gsm8k": correct_count / total_count}
    

class AQuAEvaluator(CausalLMEvaluator):
    
    def compare(self, example, raw_generation):
        # check if generation is correct
        answer = example["answer"]
        generation = extract_answer_letter(raw_generation)
        is_correct = generation.strip() == answer.strip()
        return is_correct, {"generation": generation, "answer": answer, "is_correct": is_correct}
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/AQuA": correct_count / total_count}


class SVAMPEvaluator(GSM8kEvaluator):
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/SVAMP": correct_count / total_count}
    

class MAWPSEvaluator(GSM8kEvaluator):
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/MAWPS": correct_count / total_count}

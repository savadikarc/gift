from .evaluator_base import CausalLMEvaluator


class Commonsense170kEvaluator(CausalLMEvaluator):

    def compare(self, example, raw_generation):
        # check if generation is correct
        answer = example["answer"]
        generation = raw_generation[:]
        return generation.strip() == answer.strip(), {"generation": generation, "answer": answer}
    
    def calculate_metrics(self, correct_count, total_count):
        return {f"eval/{self.dataset_name}": correct_count / total_count}

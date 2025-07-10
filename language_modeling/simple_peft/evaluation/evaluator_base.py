from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from datasets import Dataset
from tqdm import tqdm
import torch


def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.rfind(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
    return output


def make_data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return data_collator


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class Evaluator:

    def __init__(
            self, 
            dataset_name,
            model,
            tokenizer: AutoTokenizer,
            eval_dataset: Dataset,
            data_items: list, # raw_dataset
            trigger_tokens: str, # For extracting answer from the prediction
            label_trigger_tokens: str, # For extracting answer from the gt labels
            query_field: str,
            response_field: str,
            batch_size: int=4,
            data_collator=None,
            split=None,
            device="cuda",
    ):
        self.dataset_name = dataset_name
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_items = data_items
        self.trigger_tokens = trigger_tokens
        self.label_trigger_tokens = label_trigger_tokens
        self.query_field = query_field
        self.response_field = response_field
        self.batch_size = batch_size
        self.data_collator = data_collator
        self.split = split
        self.device = device

        self.eval_dataloader = self.make_dataloader()

    def make_dataloader(self):
        raise NotImplementedError

    def generate(self):
        # Override this method in subclasses
        raise NotImplementedError

    def compare(self, example, raw_generation):
        raise NotImplementedError
    
    def calculate_metrics(self, predicted, labels):
        raise NotImplementedError

    def evaluate():
        raise NotImplementedError
    

class CausalLMEvaluator(Evaluator):

    def make_dataloader(self):
        self.tokenizer.padding_side = "left" # switch padding side for collator

        data_collator = self.data_collator if self.data_collator is not None else \
            make_data_collator(self.tokenizer, self.model)
        eval_dataloader = make_dataloader(self.eval_dataset, self.batch_size, data_collator, shuffle=False)

        return eval_dataloader
    
    def generate(self, inputs, terminators=None, **generation_args):
        """Boilerplate code for Causal generation. Change if needed."""

        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        # set generation args depending on task
        model_args = {
            "input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
            "eos_token_id": self.tokenizer.eos_token_id,
            "early_stopping": True,
        }

        model_args.update(generation_args)
        if "Meta-Llama-3-8B-Instruct" in self.tokenizer.name_or_path: # pretty bad workaround for llama-3, forgive me
            model_args["eos_token_id"] = terminators

        with torch.no_grad():    
            output = self.model.generate(**model_args)

        # detokenize in batch
        actual_preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        
        return actual_preds
    
    def extract_output(self, pred, trigger_tokens=None):
        if trigger_tokens:
            return extract_output(pred, trigger_tokens)
        return pred
    
    def evaluate(self, **generation_args):
        
        """Boilerplate code for Causal generation tasks. Change if needed."""

        correct_count = 0
        total_count = 0
        generations = []
        eval_iterator = tqdm(self.eval_dataloader, position=0, leave=True)

        terminators = None
        trigger_tokens = self.trigger_tokens

        for step, inputs in enumerate(eval_iterator):
            
            ## Step 1: Generate on a batch
            actual_preds = self.generate(inputs, terminators, **generation_args)

            for id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = self.data_items[id]
                is_valid = True
                
                # Step 2: Extract output for each item in batch
                raw_generation = self.extract_output(pred, trigger_tokens)
                # print("Pred", pred)
                # print("Raw Generation", raw_generation)
                # print("-------------------")
                if not raw_generation:
                    is_valid = False
                    raw_generation = pred
                    print("get not split based on trigger tokens: ", pred)

                generation_logs = {"raw_generation": raw_generation, "answer": example["answer"]}
                if is_valid:
                    # Step 3: check if generation is correct
                    is_correct, answer_logs = self.compare(example, raw_generation)
                    if is_correct:
                        correct_count += 1
                    generation_logs.update(answer_logs)
                        
                # log
                total_count += 1
                metric_str = round(correct_count / total_count, 3)
                eval_iterator.set_postfix({"em": metric_str})
                instruction = example[self.eval_dataset.query_field]
                generations += [{
                    "valid": is_valid,
                    "instruction": instruction,
                    **generation_logs
                }]

        return generations, self.calculate_metrics(correct_count, total_count)

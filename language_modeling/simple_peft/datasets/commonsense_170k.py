import os
from math import floor

import transformers
import datasets
from datasets import load_dataset, concatenate_datasets

from .datasets_base import LMDataset, JSONDataset
from ..constants import IGNORE_INDEX

__all__ = [
    "Commonsense170K",
    "BoolQ",
    "PiQA",
    "SocialIQA",
    "Hellaswag",
    "Winogrande",
    "ARCEasy",
    "ARCChallenge",
    "OpenBookQA"
]


class Commonsense170K(LMDataset):
    dataset_name = "commonsense_170k"

    def __init__(
            self, 
            tokenizer: transformers.PreTrainedTokenizer,
            query_field: str, 
            response_field: str,
            trigger_tokens: str,
            prompt_template: str,
            data_path: str=None,
            data_split="train", 
            dataset=None, 
            seed=42, 
            max_percent_example=None,
            ignore_index: int = IGNORE_INDEX,
            validation_percent=0.1,
            **kwargs):
        
        self.constituent_datasets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
        self.max_percent_example = max_percent_example
        super().__init__(
            tokenizer, 
            query_field, 
            response_field, 
            trigger_tokens, 
            prompt_template, 
            data_path=data_path, 
            data_split=data_split, 
            dataset=dataset, 
            seed=seed, 
            max_n_example=None, 
            ignore_index=ignore_index, 
            validation_percent=validation_percent,
            **kwargs
        )

    def load_dataset(self, **kwargs):
        task_datasets = []
        for dataset_name in self.constituent_datasets:
            data_path = os.path.join(self.data_path, dataset_name, f"{self.data_loading_split}.json")
            task_dataset = load_dataset("json", data_files=data_path, split="train")
            task_dataset = self.create_split(task_dataset, **kwargs)
            if self.max_percent_example is not None:
                max_n_example = int(len(task_dataset) * self.max_percent_example)
                task_dataset = task_dataset.shuffle(seed=self.seed)
                task_dataset = task_dataset.select(range(max_n_example))
            task_datasets.append(task_dataset)
        task_dataset = concatenate_datasets(task_datasets)

        self.raw_dataset = task_dataset
        return task_dataset

    def create_split(self, dataset: datasets.Dataset=None, **kwargs):
        assert self.data_split in ["train", "validation", "test"], f"Invalid data split: {self.data_split}"
        if self.data_split == "test":
            return dataset
        
        total_train_samples = len(dataset)
        
        if self.validation_percent is not None and self.validation_percent > 0:
            split_samples = floor(total_train_samples * self.validation_percent)

            start = 0 if self.data_split == "train" else total_train_samples - split_samples
            end = total_train_samples - split_samples if self.data_split == "train" else total_train_samples
            task_dataset = dataset.select(range(start, end))
        else:
            task_dataset = dataset
        
        # Update the raw_dataset to reflect the new split
        self.raw_dataset = task_dataset
        return task_dataset

    def postprocess(self, dataset: datasets.Dataset = None, **kwargs):
        # Do nothing. Create splits in load_dataset since we will need stratified splits.
        return


class BoolQ(JSONDataset):
    dataset_name = "boolq"


class PiQA(JSONDataset):
    dataset_name = "piqa"


class SocialIQA(JSONDataset):
    dataset_name = "social_i_qa"


class Hellaswag(JSONDataset):
    dataset_name = "hellaswag"


class Winogrande(JSONDataset):
    dataset_name = "winogrande"


class ARCEasy(JSONDataset):
    dataset_name = "ARC-Easy"


class ARCChallenge(JSONDataset):
    dataset_name = "ARC-Challenge"


class OpenBookQA(JSONDataset):
    dataset_name = "openbookqa"

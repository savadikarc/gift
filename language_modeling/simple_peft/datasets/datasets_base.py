import os
import abc
from copy import deepcopy
from collections import defaultdict
from typing import Dict
from math import floor, ceil

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import transformers
import datasets
from datasets import load_dataset
from datasets import Dataset as HFDataset

from ..constants import IGNORE_INDEX


class LMDataset(Dataset):
    __metaclass__ = abc.ABCMeta
    dataset_name = None

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
        max_n_example=None,
        ignore_index: int = IGNORE_INDEX,
        validation_percent=None,
        cache_dir=None,
        **kwargs,
    ):
        super(LMDataset, self).__init__()
        result = defaultdict(list)

        # setup
        self.tokenizer = tokenizer
        self.query_field = query_field
        self.response_field = response_field
        self.trigger_tokens = trigger_tokens
        self.data_path = data_path
        self.data_split = data_split
        self.data_loading_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.validation_percent = validation_percent

        self.ignore_index = ignore_index

        self.pad_mode = "first"
        self.fields_to_pad = ["input_ids", "labels"]
        self.fields_to_mask = ["input_ids"]
        self.prompt_template = prompt_template

        self.cache_dir = cache_dir
        self.create_dataset(**kwargs)

    @property
    def cache_exists(self):
        cache_name = f"{self.dataset_name}_{self.data_split}_{self.tokenizer.name_or_path}"
        tokenized_cache_path = os.path.join(self.cache_dir, cache_name)
        raw_cache_path = os.path.join(self.cache_dir, f"{cache_name}_raw")
        return os.path.exists(tokenized_cache_path) and os.path.exists(raw_cache_path)

    def cache_dataset(self, cache_dir):
        # Save the tokenized dataset to cache using hf datasets functionality
        cache_name = f"{self.dataset_name}_{self.data_split}_{self.tokenizer.name_or_path}"
        tokenized_cache_path = os.path.join(cache_dir, cache_name)
        self.result.save_to_disk(tokenized_cache_path)

        # Save the raw dataset to cache using hf datasets functionality
        if self.raw_dataset is not None:
            raw_cache_path = os.path.join(cache_dir, f"{cache_name}_raw")
            self.raw_dataset.save_to_disk(raw_cache_path)

    def load_from_cache(self, cache_dir):
        cache_name = f"{self.dataset_name}_{self.data_split}_{self.tokenizer.name_or_path}"
        print(f"Loading dataset from cache: {cache_name}")
        # Load the tokenized dataset from cache
        tokenized_cache_path = os.path.join(cache_dir, cache_name)
        self.result = HFDataset.load_from_disk(tokenized_cache_path)

        print(f"Loading raw dataset from cache: {cache_name}_raw")
        # Load the raw dataset from cache using hf datasets functionality
        raw_cache_path = os.path.join(cache_dir, f"{cache_name}_raw")
        self.raw_dataset = HFDataset.load_from_disk(raw_cache_path)

    def create_dataset(self, **kwargs):

        # check if cache exists
        if self.cache_dir is not None and self.cache_exists:
            self.load_from_cache(self.cache_dir)
            return

        self.preprocess(**kwargs)
        self.task_dataset = self.load_dataset(**kwargs)

        # kwargs settings
        self.postprocess(**kwargs)

        # tokenize
        # self.fields_to_remove = self.task_dataset.column_names # By default, remove these in __getitem__
        self.result = self.tokenize_dataset(self.task_dataset)

        # cache the dataset
        if self.cache_dir is not None:
            self.cache_dataset(self.cache_dir)
    
    def map(self, data_item, idx):
        tokenized = self.tokenize(data_item)
        tokenized = self.compute_padding_and_masks(idx, tokenized)
        return tokenized
    
    def preprocess_response(self, response):
        return response

    def preprocess(self, **kwargs):
        # basic setup
        self.num_labels = None
        if self.data_split in ["validation", "tune"]:
            self.data_loading_split = "train"
        print(f"Preprocessing {self.data_split} split, loading {self.data_loading_split} split")

    def tokenize_dataset(self, dataset: HFDataset):
        # tokenize
        fields_to_remove = self.task_dataset.column_names # By default, remove these in __getitem__
        return dataset.map(self.map, with_indices=True, remove_columns=fields_to_remove, load_from_cache_file=False)

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

    def postprocess(self, **kwargs):
        """Postprocessing. Default behavior is to create splits. Override this function for custom splits"""
        self.task_dataset = self.create_split(self.task_dataset, **kwargs)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return deepcopy(self.result[i])
    
    def subsample_dataset(self, dataset, n_samples):
        dataset = dataset.shuffle(seed=self.seed)
        dataset = dataset.select(range(n_samples))
        return dataset

    def load_dataset(self, **kwargs):
        """Load the dataset (or a portion of it) from HF or a local file."""

        # load the dataset
        if self.dataset is None:
            print("loading data for dataset: ", self.data_path)
            data_loading_split = self.data_loading_split
            print(f"{data_loading_split} split")
            if self.data_path is None:
                task_dataset = load_dataset(self.dataset_name, split=data_loading_split)
            elif self.data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=self.data_path, split="train")
            else:
                print("loading data for dataset: ", self.data_path)
                task_dataset = load_dataset(self.dataset_name, self.data_path, split=data_loading_split)
        else:
            task_dataset = self.dataset

        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = self.subsample_dataset(task_dataset, self.max_n_example)

        self.raw_dataset = task_dataset
        return task_dataset
    
    def compute_padding_and_masks(self, id: int, result: dict):
        result["id"] = id
            
        # add a single padding token BEFORE input_ids and fix everything
        if self.pad_mode == "first":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((torch.tensor([self.ignore_index,]), result[field]))
                else:
                    result[field] = torch.cat((torch.tensor([self.tokenizer.pad_token_id,]), result[field]))
        elif self.pad_mode == "last":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((result[field], torch.tensor([self.ignore_index,])))
                else:
                    result[field] = torch.cat((result[field], torch.tensor([self.tokenizer.pad_token_id,])))
        
        # attention masks
        if len(self.fields_to_mask) == 1:
            result["attention_mask"] = (result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id).int()
        else:
            for field in self.fields_to_mask:
                result[f"{field}_mask"] = (result[field] != self.tokenizer.pad_token_id).int()

        return result

    def tokenize(self, data_item):
        result = {}

        # set up prompt
        base_prompt = self.prompt_template % (data_item[self.query_field])
        # note: we remove the extra space here to keep the format clean.
        response = self.preprocess_response(data_item[self.response_field])
        base_input = base_prompt + f"{response}{self.tokenizer.eos_token}"
        # tokenize
        base_prompt_ids = self.tokenizer(
            base_prompt, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.data_split in ["train", "tune"]:
            base_input_ids = self.tokenizer(
                base_input, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]

            output_ids = deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = self.ignore_index
                
            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            # validation or test split
            result["input_ids"] = base_prompt_ids

        return result
    

class JSONDataset(LMDataset):

    def load_dataset(self, **kwargs):
        """Load the dataset (or a portion of it) from a local file."""

        # load the dataset
        if self.dataset is None:
            data_path = os.path.join(self.data_path, self.dataset_name, f"{self.data_loading_split}.json")
            print("loading data for dataset: ", data_path)
            task_dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            task_dataset = self.dataset

        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))

        self.raw_dataset = task_dataset
        return task_dataset

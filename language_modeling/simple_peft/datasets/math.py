from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset as HFDataset
import transformers

from .datasets_base import LMDataset, JSONDataset
from ..constants import IGNORE_INDEX


class MetaMathQA(LMDataset):
    dataset_name = "meta-math/MetaMathQA"

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer, 
            query_field: str, 
            response_field: str, 
            trigger_tokens: str,
            prompt_template: str, 
            data_path: str = None, 
            data_split="train", 
            dataset=None, 
            seed=42, 
            max_n_example=None, 
            ignore_index: int = IGNORE_INDEX, 
            validation_percent=None, 
            cache_dir=None, 
            max_len=512,
            **kwargs):
        self.max_len = max_len
        super().__init__(tokenizer, query_field, response_field, trigger_tokens, prompt_template, data_path, data_split, dataset, seed, max_n_example, ignore_index, validation_percent, cache_dir, **kwargs)

    def load_dataset(self, **kwargs):
        print("loading data for dataset: ", self.data_path)
        data_loading_split = self.data_loading_split
        print(f"{data_loading_split} split")
        
        dataset = load_dataset(self.dataset_name, split=data_loading_split)
        dataset.shuffle(seed=self.seed)

        samples = []

        bar = tqdm(dataset, total=100000)
        total = 0
        count = 0
        len_discard = 0
        for sample in dataset:
            if count >= 100000:
                break

            total += 1
            if "GSM" not in sample["type"]:
                continue # Skip MATH questions

            tokenized_text = self.tokenize(sample)["input_ids"]
            if len(tokenized_text) > self.max_len:
                len_discard += 1
                continue
            
            count += 1
            bar.update(1)
            bar.set_description(f"accepted: {count}, len discard: {len_discard}, total seen: {total}")
            samples.append(sample)

        task_dataset = HFDataset.from_list(samples)
        self.raw_dataset = task_dataset
        return task_dataset


class GSM8K(LMDataset):
    dataset_name = "openai/gsm8k"

    def preprocess_response(self, response):
        return response.replace("####", "The answer is:")

    def load_dataset(self, **kwargs):
        data_loading_split = self.data_loading_split
        print(f"{data_loading_split} split")
        
        dataset = load_dataset(self.dataset_name, "main", split=data_loading_split)
        if self.data_split == "train":
            dataset.shuffle(seed=self.seed)

        self.raw_dataset = dataset
        return dataset


class Math10k(JSONDataset):
    dataset_name = "math_10k"


class GSM8kJSON(JSONDataset):
    dataset_name = "gsm8k"


class SVAMP(JSONDataset):
    dataset_name = "SVAMP"

    
class MAWPS(JSONDataset):
    dataset_name = "mawps"

class AQuA(JSONDataset):
    dataset_name = "AQuA"
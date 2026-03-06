import json
import os
from typing import TYPE_CHECKING, Callable, Optional

import datasets

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

MAPPING = {
    "gsm8k": "openai_gsm8k_{}.json",
    "spider": "xlangai_spider_{}.json",
    "finance": "gbharti_finance-alpaca_{}.json",
    "code": "code_python_{}.json",
    "piqa": "ybisk_piqa_{}.json",
    "mbpp": "google-research-datasets_mbpp_{}.json",
    "mt_bench": "HuggingFaceH4_mt_bench_prompts_{}.json",
    "human_eval": "openai_openai_humaneval_{}.json",
}

DATA_PATH = "/export/home/lanliwei.1/code/SpecInfer/data"


def get_default_process_fn(
    tokenizer: "PreTrainedTokenizerBase"
):
    def process_fn(
        batch
    ):
        conversation = batch["conversation"]
        messages = [
            i[:-1] for i in conversation
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return process_fn


class DatasetBatchIterator:
    def __init__(self, dataset, batch_size: int, process_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.process_fn = process_fn

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i: i + self.batch_size]
            if self.process_fn:
                yield self.process_fn(batch)
            else:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DatasetLoader:
    @classmethod
    def load_dataset(
        cls,
        name: str,
        path: str = DATA_PATH,
        split: str = "train",
    ) -> list[dict]:
        if name not in MAPPING.keys():
            raise ValueError(f"Dataset {name} not supported.")
        file = MAPPING[name].format(split)
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    @classmethod
    def load_iter_dataset(
        cls,
        name: str,
        path: str = DATA_PATH,
        split: str = "train",
        batch_size: int = 1,
        process_fn: Optional[Callable] = None,
    ):
        data = cls.load_dataset(name, path, split)
        dataset = datasets.Dataset.from_list(data)
        return DatasetBatchIterator(dataset, batch_size, process_fn)

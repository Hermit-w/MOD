import json
import os
import random
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
    "cnn_dm": "abisee_cnn_dailymail_{}.json",
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
    def __init__(self, dataset, batch_size: int, process_fn: Optional[Callable] = None, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.process_fn = process_fn
        self.drop_last = drop_last

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i: i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            if self.process_fn:
                yield self.process_fn(batch)
            else:
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MixedDataset:
    def __init__(
        self,
        dataset_names: list[str],
        split: str = "train",
        length_per_dataset: Optional[int] = None,
        sample_each_dataset: bool = False,
        shuffle: bool = True,
        batch_size: int = 1,
        process_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        self.data = []
        for name in dataset_names:
            dataset_data = DatasetLoader.load_dataset(name, split=split)

            if length_per_dataset is not None:
                if len(dataset_data) > length_per_dataset:
                    if sample_each_dataset:
                        dataset_data = random.sample(dataset_data, length_per_dataset)
                    else:
                        dataset_data = dataset_data[:length_per_dataset]

            self.data.extend(dataset_data)

        if shuffle:
            random.shuffle(self.data)

        self.dataset = datasets.Dataset.from_list(self.data)
        self.batch_iterator = DatasetBatchIterator(self.dataset, batch_size, process_fn, drop_last)

    @property
    def process_fn(self):
        return self.batch_iterator.process_fn

    @process_fn.setter
    def process_fn(self, fn: Callable):
        self.batch_iterator.process_fn = fn

    def __len__(self):
        return len(self.batch_iterator)

    def __iter__(self):
        return iter(self.batch_iterator)


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
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file {file_path} not found."
                f"Please make sure the dataset is extracted and placed in the correct path."
                f"Or {name} dataset doesn't have {split} split."
            )
        return data

    @classmethod
    def load_mixed_dataset(
        cls,
        dataset_names: list[str],
        split: str = "train",
        length_per_dataset: Optional[int] = None,
        sample_each_dataset: bool = False,
        shuffle: bool = True,
        batch_size: int = 1,
        process_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ) -> MixedDataset:
        return MixedDataset(
            dataset_names=dataset_names,
            split=split,
            length_per_dataset=length_per_dataset,
            sample_each_dataset=sample_each_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            process_fn=process_fn,
            drop_last=drop_last
        )

    @classmethod
    def load_iter_dataset(
        cls,
        name: str,
        path: str = DATA_PATH,
        split: str = "train",
        batch_size: int = 1,
        process_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        data = cls.load_dataset(name, path, split)
        dataset = datasets.Dataset.from_list(data)
        return DatasetBatchIterator(dataset, batch_size, process_fn, drop_last)

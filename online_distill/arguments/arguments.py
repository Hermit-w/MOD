import os
from dataclasses import dataclass, field

import yaml
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DatasetsArguments:
    datasets: list[str] = field(
        default_factory=list, metadata={"help": "List of datasets to use for training."}
    )
    datasets_split: str = field(
        default="train", metadata={"help": "Split of the datasets to use."}
    )
    num_samples_per_dataset: int = field(
        default=None, metadata={
            "help": "Number of samples to use from each dataset. If not set, use the entire dataset."
        }
    )
    drop_last: bool = field(
        default=False, metadata={"help": "Whether to drop the last incomplete batch."}
    )
    enable_shuffle_datasets: bool = field(
        default=False, metadata={"help": "Whether to shuffle the datasets."}
    )
    random_seed: int = field(
        default=42, metadata={"help": "Random seed for shuffling the datasets."}
    )


@dataclass
class OnlineDistillArguments:
    model_name: str = field(
        metadata={"help": "Path to the original model or model identifier from huggingface.co/models."}
    )
    draft_model_name: str = field(
        metadata={"help": "Path to the draft model or model identifier from huggingface.co/models."}
    )
    num_gpus_train: int = field(
        default=1, metadata={"help": "Number of GPUs to use for training."}
    )
    num_gpus_inference: int = field(
        default=1, metadata={"help": "Number of GPUs to use for vLLM inference."}
    )
    num_gpus_transformer: int = field(
        default=1, metadata={"help": "Number of GPUs to use for Transformers inference."}
    )
    num_speculative_tokens: int = field(
        default=5, metadata={"help": "Number of speculative tokens to generate."}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size for inference."}
    )
    kl_loss_ratio: float = field(
        default=1.0, metadata={"help": "The ratio of KL loss in the total loss."}
    )
    loss_on_wrong_tokens: bool = field(
        default=True, metadata={"help": "Whether to calculate loss on wrong tokens."}
    )
    max_tokens: int = field(
        default=128, metadata={"help": "Maximum number of tokens to generate."}
    )
    buffer_size_threshold: int = field(
        default=16, metadata={"help": "Buffer size threshold for triggering training."}
    )
    enable_multi_drafters: bool = field(
        default=False, metadata={"help": "Whether to enable multiple drafters."}
    )
    enable_online_update: bool = field(
        default=False, metadata={"help": "Whether to enable online update."}
    )
    dry_run: bool = field(
        default=False, metadata={"help": "Whether to run in dry-run mode (no actual training or inference)."}
    )


def parse_args(args=None) -> tuple[DatasetsArguments, OnlineDistillArguments, TrainingArguments]:
    parser = HfArgumentParser((DatasetsArguments, OnlineDistillArguments, TrainingArguments))
    if args is not None:
        datasets_args, online_distill_args, training_args = parser.parse_args_into_dataclasses(args=args)
    else:
        datasets_args, online_distill_args, training_args = parser.parse_args_into_dataclasses()
    return datasets_args, online_distill_args, training_args


def save_args(
    args: tuple[DatasetsArguments, OnlineDistillArguments, TrainingArguments]
):
    datasets_args, online_distill_args, training_args = args
    args_dict = dict(
        datasets_args=datasets_args.__dict__,
        online_distill_args=online_distill_args.__dict__,
        training_args=training_args.__dict__,
    )

    save_dir = training_args.output_dir
    args_file = os.path.join(save_dir, "args.yaml")
    with open(args_file, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

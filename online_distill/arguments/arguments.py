from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments


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
    num_training_steps: int = field(
        default=1000, metadata={"help": "Number of training steps."}
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


def parse_args(args=None):
    parser = HfArgumentParser((OnlineDistillArguments, TrainingArguments))
    if args is not None:
        online_distill_args, training_args = parser.parse_args_into_dataclasses(args=args)
    else:
        online_distill_args, training_args = parser.parse_args_into_dataclasses()
    return online_distill_args, training_args

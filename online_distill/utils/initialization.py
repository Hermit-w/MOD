import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from vllm.config import CompilationMode, WeightTransferConfig

from online_distill.inference.vllm_inference import MyLLM
from online_distill.trainer.distill_trainer import DistillTrainer


def get_inference_instance(
    model_name: str,
    draft_model_name: str,
    num_gpus_inference: int,
    num_speculative_tokens: int,
):
    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * num_gpus_inference)
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    _MyLLM = MyLLM.options(num_cpus=0, num_gpus=0, scheduling_strategy=scheduling_inference)

    speculative_config = {
        "method": "draft_model",
        "model": draft_model_name,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": num_gpus_inference,
    }
    llm = _MyLLM.remote(
        model=model_name,
        compilation_config={"mode": CompilationMode.DYNAMO_TRACE_ONCE},
        speculative_config=speculative_config,
        disable_log_stats=False,
        tensor_parallel_size=num_gpus_inference,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    return llm


def get_training_instance(
    model_name: str,
    num_gpus_train: int,
    training_args: TrainingArguments,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _DistillTrainer = DistillTrainer.options(num_gpus=num_gpus_train)
    trainer = _DistillTrainer.remote(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
    )

    return trainer


def init_all(
    model_name: str,
    draft_model_name: str,
    num_gpus_train: int,
    num_gpus_inference: int,
    num_speculative_tokens: int,
    training_args: TrainingArguments,
):
    ray.init()

    llm = get_inference_instance(
        model_name=model_name,
        draft_model_name=draft_model_name,
        num_gpus_inference=num_gpus_inference,
        num_speculative_tokens=num_speculative_tokens
    )

    trainer = get_training_instance(
        model_name=draft_model_name,
        num_gpus_train=num_gpus_train,
        training_args=training_args,
    )

    # Get trainer address
    master_address, master_port = ray.get(trainer.get_master_address_and_port.remote())

    # Calculate world size
    world_size = ray.get(llm.get_world_size.remote()) + 1

    inference_handle = llm.init_weight_transfer_engine.remote(
        dict(
            init_info=dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )
    )

    train_handle = trainer.init_weight_transfer_group.remote(world_size)

    ray.get([train_handle, inference_handle])

    return llm, trainer

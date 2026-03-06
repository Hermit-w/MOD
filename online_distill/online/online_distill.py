import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from vllm import SamplingParams
from vllm.config import CompilationMode, WeightTransferConfig

from ..inference.transformers_inference import MyTransformer
from ..inference.vllm_inference import MyLLM
from ..trainer.distill_trainer import DistillTrainer
from ..utils.update_weight import update_drafter_weights

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import RequestOutput
    from vllm.v1.metrics.reader import Metric


@dataclass
class DistillSample:
    prompt_token_id: torch.Tensor
    output_token_id: torch.Tensor
    output_logprobs: torch.Tensor
    loss_mask: torch.Tensor

    @classmethod
    def from_request_output(
        cls,
        request_output: "RequestOutput",
        vocab_size: int,
        only_loss_on_wrong_ids: bool = True,
    ) -> "DistillSample":
        prompt_token_id = torch.tensor(request_output.prompt_token_ids, dtype=torch.int64)
        output_token_id = torch.tensor(request_output.outputs[0].token_ids, dtype=torch.int64)
        origin_logprobs = request_output.outputs[0].logprobs
        output_logprobs = torch.full(
            (len(output_token_id), vocab_size),
            fill_value=-9999.0,
            dtype=torch.float32,
        )
        assert origin_logprobs is not None, "Logprobs should not be None when creating DistillRequest"
        for i, logprobs in enumerate(origin_logprobs):
            for token_id, logprob_obj in logprobs.items():
                output_logprobs[i, token_id] = logprob_obj.logprob

        history_token_ids: list[list[int]] = request_output.outputs[0].history_token_ids
        loss_mask = torch.ones_like(output_token_id, dtype=torch.bool)
        if only_loss_on_wrong_ids:
            loss_mask.zero_()  # set all to 0 first
            sum_count = 0
            for history_token_id in history_token_ids:
                count = len(history_token_id)
                sum_count += count
                loss_mask[sum_count - 1] = 1  # only set the last token of each draft to 1
            loss_mask[0] = 0  # the first token is always correct, so set it to 0
        return cls(
            prompt_token_id=prompt_token_id,
            output_token_id=output_token_id,
            output_logprobs=output_logprobs,
            loss_mask=loss_mask,
        )


@dataclass
class DistillBuffer:
    prototype_data: torch.Tensor
    request_buffer: list[DistillSample]


class OnlineDistillWorker:
    def __init__(
        self,
        model_name: str,
        draft_model_name: str,
        num_gpus_train: int,
        num_gpus_inference: int,
        num_gpus_transformer: int,
        num_speculative_tokens: int,
        training_args: TrainingArguments,
        num_training_steps: int,
        max_tokens: int = 128,
        buffer_size_threshold: int = 16,
        enable_multi_drafters: bool = False,
        enable_online_update: bool = False,
    ):
        self.model_name = model_name
        self.draft_model_name = draft_model_name
        self.num_speculative_tokens = num_speculative_tokens
        self.num_training_steps = num_training_steps
        self.buffer_size_threshold = buffer_size_threshold
        self.enable_multi_drafters = enable_multi_drafters
        self.enable_online_update = enable_online_update

        self._cache_metrics: dict[str, int | list[int]] | None = None
        self.sample_buffer: list[DistillBuffer] = []
        self.alpha_per_pos = [[] for _ in range(num_speculative_tokens)]
        self.alphas = []
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            top_k=1,
            logprobs=20,
            include_stop_str_in_output=True,
        )

        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        ray.init()
        self.tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(model_name)
        self.trainer = self._get_training_instance(
            model_name=draft_model_name,
            num_gpus_train=num_gpus_train,
            training_args=training_args,
        )
        self.llm = self._get_inference_instance(
            model_name=model_name,
            draft_model_name=draft_model_name,
            num_gpus_inference=num_gpus_inference,
            num_speculative_tokens=num_speculative_tokens,
        )
        self.transformer = self._get_transformers_inference_instance(
            model_name=model_name,
            num_gpus_inference=num_gpus_transformer,
        )
        # prepare for weight transfer
        self._init_weight_engine()

    def _init_weight_engine(self):
        master_address, master_port = ray.get(self.trainer.get_master_address_and_port.remote())
        world_size = ray.get(self.llm.get_world_size.remote()) + 1
        inference_handle = self.llm.init_weight_transfer_engine.remote(
            dict(
                init_info=dict(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                )
            )
        )

        train_handle = self.trainer.init_weight_transfer_group.remote(world_size)

        ray.get([train_handle, inference_handle])

    def _get_inference_instance(
        self,
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
            enable_chunked_prefill=False,
            disable_log_stats=False,
            tensor_parallel_size=num_gpus_inference,
            distributed_executor_backend="ray",
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )
        return llm

    def _get_training_instance(
        self,
        model_name: str,
        num_gpus_train: int,
        training_args: TrainingArguments,
    ):
        if not self.enable_online_update:
            return None

        def model_init():
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto"
            )
        tokenizer = self.tokenizer
        _DistillTrainer = DistillTrainer.options(num_gpus=num_gpus_train)
        trainer = _DistillTrainer.remote(
            model_init=model_init,
            tokenizer=tokenizer,
            args=training_args,
        )
        trainer.init_training.remote(self.num_training_steps)
        return trainer

    def _get_transformers_inference_instance(
        self,
        model_name: str,
        num_gpus_inference: int,
    ):
        if not self.enable_online_update or not self.enable_multi_drafters:
            return None
        _MyTransformer = MyTransformer.options(num_gpus=num_gpus_inference)
        transformer = _MyTransformer.remote(model_name_or_path=model_name)
        return transformer

    def step(
        self,
        prompts: list[list[dict[str, str]]] | list[str]
    ):
        """
        This function will first call vllm to run the prompts.
        """
        _prompts: list[str] = prompts
        if not isinstance(prompts[0], str):
            _prompts = self.tokenizer.apply_chat_template(
                conversation=prompts,
                tokenize=False,
                add_generation_prompt=True,
            )

        outputs: list["RequestOutput"] = ray.get(self.llm.generate.remote(_prompts, self.sampling_params))
        metrics: list["Metric"] = ray.get(self.llm.get_metrics.remote())
        # collect what we need here
        # metrics related to speculative decoding
        self._update_spec_metrics(metrics)
        if self.enable_online_update:
            self._update_sample_buffer(outputs)
            self._maybe_do_training()

    def _update_spec_metrics(self, metrics: list["Metric"]):
        # vllm:spec_decode_num_drafts: 47
        # vllm:spec_decode_num_draft_tokens: 235
        # vllm:spec_decode_num_accepted_tokens: 82
        # vllm:spec_decode_num_accepted_tokens_per_pos: [35, 22, 10, 9, 6]
        spec_metrics: dict[str, int | list[int]] = {}
        for metric in metrics:
            if "spec" in metric.name:
                try:
                    value = metric.value
                except Exception:
                    value = metric.values
                spec_metrics[metric.name] = value
        last_num_drafts = 0
        last_num_draft_tokens = 0
        last_num_accepted_tokens = 0
        last_num_accepted_tokens_per_pos = [0 for _ in range(self.num_speculative_tokens)]
        if self._cache_metrics is not None:
            last_num_drafts = self._cache_metrics["vllm:spec_decode_num_drafts"]
            last_num_draft_tokens = self._cache_metrics["vllm:spec_decode_num_draft_tokens"]
            last_num_accepted_tokens = self._cache_metrics["vllm:spec_decode_num_accepted_tokens"]
            last_num_accepted_tokens_per_pos = self._cache_metrics["vllm:spec_decode_num_accepted_tokens_per_pos"]
        num_drafts: int = spec_metrics["vllm:spec_decode_num_drafts"] - last_num_drafts
        num_draft_tokens: int = spec_metrics["vllm:spec_decode_num_draft_tokens"] - last_num_draft_tokens
        num_accepted_tokens: int = spec_metrics["vllm:spec_decode_num_accepted_tokens"] - last_num_accepted_tokens
        num_accepted_tokens_per_pos: list[int] = [
            current - last for current, last in zip(
                spec_metrics["vllm:spec_decode_num_accepted_tokens_per_pos"], last_num_accepted_tokens_per_pos
            )
        ]
        self.alphas.append(num_accepted_tokens / num_draft_tokens)
        for pos in range(len(self.alpha_per_pos)):
            self.alpha_per_pos[pos].append(num_accepted_tokens_per_pos[pos] / num_drafts)
        # update cached_metrics
        self._cache_metrics = spec_metrics

    def _prepare_inputs_for_trainer(self, request_buffer: list[DistillSample]):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
            else self.tokenizer.eos_token_id

        prompts = [sample.prompt_token_id for sample in request_buffer]
        output_ids = [sample.output_token_id[:-1] for sample in request_buffer]
        teacher_logprobs = [sample.output_logprobs for sample in request_buffer]
        loss_masks = [sample.loss_mask for sample in request_buffer]

        # Left pad prompts
        prompts_reversed = [p.flip(0) for p in prompts]
        padded_prompts = torch.nn.utils.rnn.pad_sequence(
            prompts_reversed, batch_first=True, padding_value=pad_token_id
        ).flip(1)

        # Right pad output inputs
        padded_output_ids = torch.nn.utils.rnn.pad_sequence(
            output_ids, batch_first=True, padding_value=pad_token_id
        )

        input_ids = torch.cat([padded_prompts, padded_output_ids], dim=1)
        attention_mask = (input_ids != pad_token_id).long()

        padded_teacher_logprobs = torch.nn.utils.rnn.pad_sequence(
            teacher_logprobs, batch_first=True, padding_value=-9999.0
        )
        padded_loss_masks = torch.nn.utils.rnn.pad_sequence(
            loss_masks, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_logprobs=padded_teacher_logprobs,
            loss_mask=padded_loss_masks,
        )

    def _update_sample_buffer(self, outputs: list["RequestOutput"]):
        samples = [
            DistillSample.from_request_output(output, vocab_size=self.tokenizer.vocab_size) for output in outputs
        ]
        if not self.enable_multi_drafters:
            self.sample_buffer.append(DistillBuffer(prototype_data=torch.tensor(0), request_buffer=samples))

    def _maybe_do_training(self):
        assert not self.enable_multi_drafters, "Multi-drafters is not supported in this version."
        assert len(self.sample_buffer) == 1, "There should be only one buffer when multi-drafters is not enabled."
        if len(self.sample_buffer[0].request_buffer) >= self.buffer_size_threshold:
            # pack the data and send to trainer
            inputs_for_trainer = self._prepare_inputs_for_trainer(self.sample_buffer[0].request_buffer)
            update_handel = self.trainer.update.remote(inputs_for_trainer)
            # clear the buffer
            self.sample_buffer[0].request_buffer = []
            ray.get(update_handel)
            self.update_weight()

    def update_weight(self):
        names, dtype_names, shapes = ray.get(self.trainer.get_weight_metadata.remote())
        inference_handle = self.llm.collective_rpc.remote(
            update_drafter_weights,
            kwargs={
                "update_info": dict(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=True,
                )
            }
        )
        train_handle = self.trainer.broadcast_weights.remote(packed=True)

        ray.get([train_handle, inference_handle])

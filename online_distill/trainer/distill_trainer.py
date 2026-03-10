import logging
import contextlib
from typing import TYPE_CHECKING, Any, Dict, override

import ray
import torch
from transformers import Trainer
from vllm.distributed.weight_transfer.nccl_engine import \
    NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class DistillTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.port = get_open_port()
        self.master_address = get_ip()

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # we pass the logits of teacher model in 'inputs' and compute KL divergence loss
        teacher_logprobs: "torch.Tensor" = inputs.pop("teacher_logprobs")
        loss_mask: "torch.Tensor" = inputs.pop("loss_mask")
        student_outputs: "CausalLMOutput" = model(**inputs)
        student_logits: "torch.Tensor" = student_outputs.logits
        # the teacher_logprobs only contains the output part of the sequence, so we need to align the student_logits with teacher_logprobs
        output_len = teacher_logprobs.shape[1]
        prompt_len = student_logits.shape[1] - output_len
        student_logits = student_logits[:, prompt_len:, :]
        assert student_logits.shape == teacher_logprobs.shape, f"Shape mismatch: {student_logits.shape} vs {teacher_logprobs.shape}"
        student_logprobs: "torch.Tensor" = torch.nn.functional.log_softmax(student_logits, dim=-1)
        kl_loss = torch.nn.functional.kl_div(
            student_logprobs,
            teacher_logprobs,
            log_target=True,
            reduction="none"
        )
        # kl_loss shape: (batch_size, output_len, vocab_size)
        # loss_mask shape: (batch_size, output_len)
        # calculate the mean kl_loss over the output tokens, and then mean over the batch
        kl_loss = (kl_loss * loss_mask.unsqueeze(-1)).sum((1, 2)) / loss_mask.sum(-1)
        kl_loss = kl_loss.mean()
        return (kl_loss, student_outputs) if return_outputs else kl_loss

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for weight transfer."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )

    def init_training(self, num_training_steps: int):
        """Initialize optimizer and scheduler."""
        self.create_optimizer_and_scheduler(num_training_steps=num_training_steps)

    def update(self, inputs: Dict[str, Any]):
        """Perform a single training step."""
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("Optimizer and scheduler not initialized. Call init_training() first.")

        cp_context, inputs = self._prepare_context_parallel_inputs(self.model, inputs)

        with cp_context():
            self.model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(self.model, inputs)

            del inputs

            kwargs = {}

            self.accelerator.backward(loss, **kwargs)

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm,
            )

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu().item()

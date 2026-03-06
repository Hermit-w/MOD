from typing import TYPE_CHECKING

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast


@ray.remote
class MyTransformer:
    def __init__(self, model_name_or_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
        )
        self.tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

    def get_router_logits(
        self,
        prompts: list[str],
    ):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)
        outputs: "MoeCausalLMOutputWithPast" = self.model(**inputs, output_router_logits=True)
        # get mean of router logits across sequence length dimension
        # we should consider padding tokens when calculating the mean
        attention_mask = inputs["attention_mask"]
        router_logits = outputs.router_logits
        expanded_attention_mask = attention_mask.unsqueeze(-1).float()
        valid_counts = expanded_attention_mask.sum(dim=1)
        valid_counts = torch.clamp(valid_counts, min=1)  # prevent division by zero

        layer_averages = []
        for router_logit in router_logits:
            probs = router_logit.softmax(dim=-1)
            masked_probs = probs * expanded_attention_mask.to(probs.device)
            sum_masked_probs = masked_probs.sum(dim=1)
            aver = sum_masked_probs / valid_counts.to(probs.device)
            # aver shape: (batch_size, num_experts)
            layer_averages.append(aver.cpu())

        ret_router_logits = torch.stack(layer_averages, dim=0).permute(1, 0, 2)
        # ret_router_logits shape: (batch_size, num_layers, num_experts)
        return ret_router_logits

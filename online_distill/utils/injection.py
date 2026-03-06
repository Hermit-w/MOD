from dataclasses import dataclass, field
from typing import Sequence, override

import numpy as np
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason
from vllm.v1.engine.detokenizer import check_stop_strings


def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
    """
    Update RequestState for the request_id by:
        1) Detokenize the new token ids incrementally.
        2) Evaluate stop criteria.

    Return matched stop string or None.
    """
    #######################
    # the place where is different with the original method
    if not hasattr(self, "history_token_ids"):
        setattr(self, "history_token_ids", [])
    self.history_token_ids.append(new_token_ids)
    #######################
    if not new_token_ids:
        # Skip detokenization if no new token ids.
        return None

    if stop_terminated and not self.include_stop_str_in_output:
        # If stop-terminated, exclude last token from detokenization
        # based on include_stop_str_in_output parameter.
        skipped_stop_token_id = new_token_ids[-1]
        new_token_ids = new_token_ids[:-1]
    else:
        skipped_stop_token_id = None

    # 1) Detokenize the new token ids incrementally.
    stop_check_offset = len(self.output_text)
    for new_token_id in new_token_ids:
        self.token_ids.append(new_token_id)
        self.output_text += self.decode_next(new_token_id)
        # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
        if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
            stop_check_offset = len(self.output_text)

    if skipped_stop_token_id is not None:
        # Cleanup after skipping detokenization.
        self.token_ids.append(skipped_stop_token_id)

    # 2) Evaluate stop strings.
    stop_string = None
    if self.stop and self.num_output_tokens() > self.min_tokens:
        stop = check_stop_strings(
            output_text=self.output_text,
            new_char_count=len(self.output_text) - stop_check_offset,
            stop=self.stop,
            include_in_output=self.include_stop_str_in_output,
        )
        if stop is not None:
            stop_string, truncate_to = stop
            if truncate_to != -1:
                self.output_text = self.output_text[:truncate_to]

    return stop_string


@dataclass
class CustomizedCompletionOutput(CompletionOutput):
    history_token_ids: Sequence[Sequence[int]] = field(default_factory=list)

    @override
    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"history_token_ids={self.history_token_ids}"
            f"routed_experts={self.routed_experts}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


CompletionOutput = CustomizedCompletionOutput


def _new_completion_output(
    self,
    token_ids: list[int],
    finish_reason: FinishReason | None,
    stop_reason: int | str | None,
    routed_experts: np.ndarray | None = None,
) -> CompletionOutput:
    assert self.detokenizer is not None
    assert self.logprobs_processor is not None
    finished = finish_reason is not None
    delta = self.output_kind == RequestOutputKind.DELTA

    # Prepare text and token_ids, based on delta mode
    text = self.detokenizer.get_next_output_text(finished, delta)
    if not delta:
        token_ids = self.detokenizer.output_token_ids

    # Prepare logprobs, based on delta mode
    logprobs = self.logprobs_processor.logprobs
    if delta and logprobs:
        logprobs = logprobs[-len(token_ids) :]

    return CompletionOutput(
        index=self.request_index,
        text=text,
        token_ids=token_ids,
        ##################################
        # the place where is different with the original method
        history_token_ids=self.detokenizer.history_token_ids,
        ##################################
        routed_experts=routed_experts,
        logprobs=logprobs,
        cumulative_logprob=self.logprobs_processor.cumulative_logprob,
        finish_reason=str(finish_reason) if finished else None,
        stop_reason=stop_reason if finished else None,
    )

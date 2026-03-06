import warnings

import vllm.outputs
import vllm.v1.engine.detokenizer
import vllm.v1.engine.output_processor

from .online.online_distill import OnlineDistillWorker
from .utils.injection import (CustomizedCompletionOutput,
                              _new_completion_output, update)


def inject_func():
    warnings.warn("We substitute some part of vllm's code by monkey patching for prototype testing.")
    # inject what we need here
    vllm.v1.engine.detokenizer.BaseIncrementalDetokenizer.update = update
    vllm.outputs.CompletionOutput = CustomizedCompletionOutput
    vllm.v1.engine.output_processor.RequestState._new_completion_output = _new_completion_output


inject_func()

__all__ = [
    "OnlineDistillWorker",
]

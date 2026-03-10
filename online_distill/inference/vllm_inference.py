import os

import ray
from vllm import LLM

from ..utils.injection import inject_func


@ray.remote
class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        tp_size = kwargs["tensor_parallel_size"]
        vllm_ray_bundle_indices = ",".join(str(i) for i in range(tp_size))
        print(vllm_ray_bundle_indices)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = vllm_ray_bundle_indices
        inject_func()
        super().__init__(*args, **kwargs)

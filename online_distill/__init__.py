from .online.online_distill import OnlineDistillWorker
from .utils.injection import inject_func

inject_func()

__all__ = [
    "OnlineDistillWorker",
]

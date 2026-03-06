import torch


def update_drafter_weights(self, update_info: dict) -> None:
    """
    Batched weight update from the trainer.

    Args:
        update_info: Dictionary containing backend-specific update info
    """
    if self.weight_transfer_engine is None:
        raise RuntimeError(
            "Weight transfer not configured. "
            "Please set weight_transfer_config to enable weight transfer."
        )

    # Parse dict into backend-specific typed dataclass
    typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)

    model = self.model_runner.drafter.model

    if typed_update_info.is_checkpoint_format:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload, initialize_layerwise_reload)

        # Use layerwise reload pattern for checkpoint format weights
        with torch.device(self.device):
            initialize_layerwise_reload(model)
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=model.load_weights,
            )
            finalize_layerwise_reload(model, self.model_config)
    else:
        # Weights are already in kernel format, copy directly
        def load_weights_direct(
            weights: list[tuple[str, torch.Tensor]],
        ) -> None:
            for name, weight in weights:
                param = model.get_parameter(name)
                param.copy_(weight)

        self.weight_transfer_engine.receive_weights(
            typed_update_info,
            load_weights=load_weights_direct,
        )
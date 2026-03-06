from online_distill.arguments.arguments import parse_args
from online_distill.online.online_distill import OnlineDistillWorker


def main():
    online_distill_args, training_args = parse_args()

    worker = OnlineDistillWorker(
        model_name=online_distill_args.model_name,
        draft_model_name=online_distill_args.draft_model_name,
        num_gpus_train=online_distill_args.num_gpus_train,
        num_gpus_inference=online_distill_args.num_gpus_inference,
        num_gpus_transformer=online_distill_args.num_gpus_transformer,
        num_speculative_tokens=online_distill_args.num_speculative_tokens,
        training_args=training_args,
        num_training_steps=online_distill_args.num_training_steps,
        max_tokens=online_distill_args.max_tokens,
        buffer_size_threshold=online_distill_args.buffer_size_threshold,
        enable_multi_drafters=online_distill_args.enable_multi_drafters,
        enable_online_update=online_distill_args.enable_online_update,
    )

    print("OnlineDistillWorker initialized successfully.")
    return worker


if __name__ == "__main__":
    main()

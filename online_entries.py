import os
import random
import time

from tqdm import tqdm

from online_distill.arguments.arguments import parse_args, save_args
from online_distill.data import DatasetLoader
from online_distill.data.loader import MixedDataset, get_default_process_fn
from online_distill.online.online_distill import OnlineDistillWorker
from online_distill.utils.logger import setup_logger


def main():
    datasets_args, online_distill_args, training_args = parse_args()
    output_dir = training_args.output_dir
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(output_dir, timestamp)
    training_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(log_path=output_dir, level="info")

    random.seed(datasets_args.random_seed)

    if online_distill_args.dry_run:
        save_args((datasets_args, online_distill_args, training_args))
        return

    datasets = DatasetLoader.load_mixed_dataset(
        dataset_names=datasets_args.datasets,
        split=datasets_args.datasets_split,
        length_per_dataset=datasets_args.num_samples_per_dataset,
        shuffle=datasets_args.enable_shuffle_datasets,
        batch_size=online_distill_args.batch_size,
        drop_last=datasets_args.drop_last,
    )

    num_training_steps = len(datasets)

    worker = OnlineDistillWorker(
        model_name=online_distill_args.model_name,
        draft_model_name=online_distill_args.draft_model_name,
        num_gpus_train=online_distill_args.num_gpus_train,
        num_gpus_inference=online_distill_args.num_gpus_inference,
        num_gpus_transformer=online_distill_args.num_gpus_transformer,
        num_speculative_tokens=online_distill_args.num_speculative_tokens,
        training_args=training_args,
        num_training_steps=num_training_steps,
        max_tokens=online_distill_args.max_tokens,
        loss_on_wrong_tokens=online_distill_args.loss_on_wrong_tokens,
        buffer_size_threshold=online_distill_args.buffer_size_threshold,
        enable_multi_drafters=online_distill_args.enable_multi_drafters,
        enable_online_update=online_distill_args.enable_online_update,
    )

    datasets.process_fn = get_default_process_fn(worker.tokenizer)

    run(datasets, worker)

    save_args((datasets_args, online_distill_args, training_args))
    worker.save_metrics()


def run(datasets: MixedDataset, worker: OnlineDistillWorker):
    for batch in tqdm(datasets.batch_iterator, desc="Processing batches"):
        worker.step(batch)


if __name__ == "__main__":
    main()

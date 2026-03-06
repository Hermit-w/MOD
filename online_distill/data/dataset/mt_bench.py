import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    INSTRUCTION = ""
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": INSTRUCTION + prompt
        }
        for prompt in case['prompt']
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "HuggingFaceH4/mt_bench_prompts"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "default")
    dataset.extract_data(dataset.dataset.keys(), transform)
    # print(dataset.get_raw_data()["train"][0])

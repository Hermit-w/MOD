import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    code_prompt = "Please generate python code based on the following requests:\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content":  code_prompt + case['text']
        },
        {
            "role": "assistant",
            "content": case['code']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "google-research-datasets/mbpp"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "full")
    dataset.extract_data(dataset.dataset.keys(), transform)

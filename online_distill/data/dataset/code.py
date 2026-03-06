import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    code_prompt = "Please generate python code based on the following doc:\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": code_prompt + case['func_documentation_string']
        },
        {
            "role": "assistant",
            "content": case['func_code_string']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "code_search_net/python/1.0.0/8f2524e6b62f65af5f5d65c53715c654db7b08dc93e0b7bcce2ab2f286a75be1"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path)
    dataset.extract_data(dataset.dataset.keys(), transform, enforce_output_name="data/code_python_train.json")

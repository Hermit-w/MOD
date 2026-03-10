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
    dataset_name = "code-search-net/code_search_net/python"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path)
    dataset.extract_data(dataset.dataset.keys(), transform, size=3000, enforce_output_name="data/code_python_{}.json")

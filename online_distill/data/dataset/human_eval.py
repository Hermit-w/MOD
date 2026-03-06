import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    INSTRUCTION = "Please complete the following function in Python.\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": INSTRUCTION + case['prompt']
        },
        {
            "role": "assistant",
            "content": case['canonical_solution']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "openai/openai_humaneval"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "openai_humaneval")
    dataset.extract_data(dataset.dataset.keys(), transform)

import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    INSTRUCTION = "Provide solution for the following goal.\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content":  INSTRUCTION + case['goal']
        },
        {
            "role": "assistant",
            "content": case['sol1'] if case["label"] == 0 else case['sol2']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "ybisk/piqa"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "default")
    dataset.extract_data(dataset.dataset.keys(), transform)

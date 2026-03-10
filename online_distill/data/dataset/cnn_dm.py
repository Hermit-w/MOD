import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    INSTRUCTION = "Please summary the following article:\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": INSTRUCTION + case['article']
        },
        {
            "role": "assistant",
            "content": case['highlights']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "abisee/cnn_dailymail"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, '3.0.0')
    dataset.extract_data(dataset.dataset.keys(), transform, 3000)

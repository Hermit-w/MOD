import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    INSTRUCTION = "You are a financial expert. Provide detailed and accurate responses to the following queries related to finance and investment.\n"
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": INSTRUCTION + case['instruction']
        },
        {
            "role": "assistant",
            "content": case['output']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "gbharti/finance-alpaca"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path)
    dataset.extract_data(dataset.dataset.keys(), transform)
    # print(dataset.get_raw_data()["train"][0])

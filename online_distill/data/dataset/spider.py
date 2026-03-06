import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    _case = {}
    SQL_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "  # noqa
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": SQL_prompt + case['question']
        },
        {
            "role": "assistant",
            "content": " ".join(case['query_toks_no_value'])
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "xlangai/spider"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path)
    dataset.extract_data(dataset.dataset.keys(), transform)

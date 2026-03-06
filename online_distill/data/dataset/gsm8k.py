import os

from dataset_wrapper import ORIGIN_DATA_PATH, DatasetWrapper


def transform(i, case):
    # INSTRUCTION = "Solve the following math problem step-by-step.\n"
    INSTRUCTION = "You are a math expert. Provide detailed and accurate solutions to the following math problems.\n"
    # INSTRUCTION = ""
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": INSTRUCTION + case['question']
        },
        {
            "role": "assistant",
            "content": case['answer']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = ORIGIN_DATA_PATH
    dataset_name = "openai/gsm8k"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "main")
    dataset.extract_data(dataset.dataset.keys(), transform)

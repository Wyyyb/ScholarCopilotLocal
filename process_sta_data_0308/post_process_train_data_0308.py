import json
import os
from datasets import load_dataset


def download_dataset(output_path='../data/cite-llm-multi-cite-train.json'):
    try:
        dataset = load_dataset("ubowang/cite-llm-multi-cite-train", split="train")
        data = []
        for item in dataset:
            data.append(item)
        # 将数据集保存为jsonl文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, indent=2))

        print(f"数据已成功下载并保存到: {output_path}")
        print(f"数据集大小: {len(dataset)} 条记录")

    except Exception as e:
        print(f"处理数据时出错: {e}")


def load_train_data():
    train_data_path = '../data/cite-llm-multi-cite-train.json'
    if not os.path.exists(train_data_path):
        download_dataset()
    with open(train_data_path, "r") as fi:
        data = json.load(fi)
    return data


def split_eval():
    eval_data_path = ""
    train_data_path = ""


def main():
    train_data = load_train_data()


main()


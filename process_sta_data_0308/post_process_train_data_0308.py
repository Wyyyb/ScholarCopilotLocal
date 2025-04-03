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


def load_eval_data():
    with open("../data/eval_re_data_1k_0225.json", "r") as fi:
        eval_data = json.load(fi)
    return eval_data


def filter_train_data(eval_data, train_data):
    train_res_data = []
    eval_id_list = []
    for each in eval_data:
        paper_id = each["paper_id"].split("-")[0]
        eval_id_list.append(paper_id)
    for each in train_data:
        paper_id = each["arxiv_id"].split("-")[0]
        if paper_id in eval_id_list:
            continue
        train_res_data.append(each)
    return train_res_data


def main():
    train_data_output_path = "../data/scholar_copilot_train_data_500k.json"
    train_data = load_train_data()
    print("ori train data number", len(train_data))
    eval_data = load_eval_data()
    print("ori eval data number", len(eval_data))
    train_res_data = filter_train_data(eval_data, train_data)
    print("len(train_res_data)", len(train_res_data))
    train_res_data = train_res_data[:500000]
    print("len(train_res_data)", len(train_res_data))
    with open(train_data_output_path, "w") as fo:
        fo.write(json.dumps(train_res_data, indent=4))


main()


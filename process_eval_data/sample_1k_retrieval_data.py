import json
import random


def main():
    with open("../local_data/multi_cite_eval.json", "r") as fi:
        data = json.load(fi)
    sample_data = []
    for each in data:
        cite_map = json.loads(each["cite_corpus_id_map"])
        ss_count = 0
        arxiv_count = 0
        for k, v in cite_map.items():
            if not v.startswith("arxiv-"):
                ss_count += 1
            else:
                arxiv_count += 1
        if arxiv_count < 11:
            continue
        if ss_count > 15:
            continue
        sample_data.append(each)
    print(len(sample_data))
    output_file = "../local_data/eval_retrieval_1k_data.json"
    with open(output_file, "w") as fo:
        fo.write(json.dumps(sample_data[:1000], indent=2))
    print(sample_data[100]["cite_corpus_id_map"])


main()


import json


def sta_corpus():
    with open("/data/yubo/ScholarCopilot/data/corpus_data_arxiv_1215.jsonl", "r") as fi:
        data = json.load(fi)
    print("len(data)", len(data))


if __name__ == "__main__":
    sta_corpus()


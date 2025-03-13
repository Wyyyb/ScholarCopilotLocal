import json


def sta_corpus():
    data = []
    with open("/data/yubo/ScholarCopilot/data/corpus_data_arxiv_1215.jsonl", "r") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    print("len(data)", len(data))


if __name__ == "__main__":
    sta_corpus()


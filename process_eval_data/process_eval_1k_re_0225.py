import json


def convert_cite_map_to_sorted_list(cite_map):
    result = []
    for key, value in cite_map.items():
        if 'multi_cite' in key:
            parts = key.replace('|>', '').replace('<|', '').split('_')
            cite_num = int(parts[2])
            sub_num = int(parts[3])
            result.append((cite_num, sub_num, value))
        elif "<|cite_" in key:
            parts = key.replace('|>', '').replace('<|', '').split('_')
            result.append((int(parts[1]), 0, value))
        else:
            print("invalid citation", key)
    result = sorted(result, key=lambda x: x[0] * 10000 + x[1])
    return result


def process_single(each, corpus_map):
    res = {}
    arxiv_id = each["arxiv_id"]
    paper = each["paper"]
    cite_map = json.loads(each["cite_corpus_id_map"])
    citation_list = convert_cite_map_to_sorted_list(cite_map)
    if paper.count("<|cite_start|>") != len(citation_list):
        print("inconsistent paper citation info")
        return None
    citation_index = 0
    valid_citation_index = 0
    stage_1_cite_map = {}
    while "<|cite_start|>" in paper:
        citation_start_index = paper.index("<|cite_start|>")
        citation_end_index = paper.index("<|cite_end|>") + len("<|cite_end|>")
        reference = paper[citation_start_index: citation_end_index]
        cite_info = citation_list[citation_index]
        ori_cite_index = cite_info[0]
        if cite_info[-1].startswith("arxiv-"):
            if cite_info[-1] not in corpus_map:
                print("fatal error, cite_info[-1] not in corpus_map", cite_info[-1])
                return None
            title = corpus_map[cite_info[-1]]["title"]
            if title not in reference:
                print("fatal error, title not in reference", title, reference)
                return None
            if cite_info[1] == 0:
                placeholder = f"<|cite_{str(ori_cite_index)}|>"
            else:
                placeholder = f"<|multi-cite_{str(ori_cite_index)}_{str(cite_info[1])}|>"
            stage_1_cite_map[placeholder] = corpus_map[cite_info[-1]]
        else:
            placeholder = ""
        paper = paper[:citation_start_index] + placeholder + paper[:citation_end_index]
        citation_index += 1
    return res


def main():
    with open("../local_data/eval_retrieval_1k_data.json", "r") as fi:
        ori_data = json.load(fi)
    corpus_data = {}
    with open("/map-vepfs/yubo/ScholarCopilot/local_data/corpus_data_arxiv_1215.jsonl", "r") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            corpus_id = curr["corpus_id"]
            corpus_data[corpus_id] = curr

    eval_data = []
    for each in ori_data:
        single_res = process_single(each, corpus_data)
        if not single_res:
            continue
        eval_data.append(single_res)


def test():
    test_map = {
        "<|multi_cite_1_1|>": "arxiv-88870",
        "<|multi_cite_1_3|>": "arxiv-78819",
        "<|multi_cite_1_4|>": "arxiv-260981",
        "<|multi_cite_1_5|>": "arxiv-88684",
        "<|multi_cite_1_6|>": "arxiv-131338",
        "<|cite_3|>": "arxiv-95397",
        "<|multi_cite_1_2|>": "arxiv-68791",
        "<|cite_4|>": "arxiv-65515",
        "<|cite_5|>": "arxiv-323564",
        "<|multi_cite_6_1|>": "arxiv-131218",
        "<|multi_cite_6_2|>": "ss-1271526",
        "<|multi_cite_6_3|>": "arxiv-199412",
        "<|multi_cite_7_1|>": "arxiv-78819",
        "<|multi_cite_7_2|>": "arxiv-88684",
        "<|cite_2|>": "arxiv-60292",
        "<|multi_cite_7_3|>": "arxiv-131338",
        "<|multi_cite_8_1|>": "arxiv-131218",
        "<|multi_cite_8_2|>": "ss-1271526",
        "<|multi_cite_8_3|>": "arxiv-199412",
        "<|cite_9|>": "arxiv-78819",
        "<|cite_10|>": "arxiv-131338"
    }

    result = convert_cite_map_to_sorted_list(test_map)
    for item in result:
        print(item)


main()


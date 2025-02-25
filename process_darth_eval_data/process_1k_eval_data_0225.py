import json
from tqdm import tqdm


def main():
    input_data_path = "/data/yubowang/arxiv-llm/local_1123/step_5_integration_1123.jsonl"
    # input_data_path = "/data/yubowang/arxiv-llm/local_1123/test_step_5.jsonl"
    res = []
    with open(input_data_path, "r") as fi:
        for line in tqdm(fi):
            curr = json.loads(line)
            bib_info = curr["bib_info"]
            paper = curr["full_intro"]
            arxiv_success_count = 0
            arxiv_fail_count = 0
            if len(paper) > 100000 or len(paper) < 5000:
                # print("len(paper)", len(paper))
                continue
            bib_info_map = {}
            for k, v in bib_info.items():
                if v["citation_corpus_id"] and v["citation_corpus_id"].startswith("arxiv-"):
                    bib_info_map[k] = v
                    arxiv_success_count += 1
                else:
                    arxiv_fail_count += 1
            curr["bib_info"] = bib_info_map
            if arxiv_success_count < 20:
                continue
            if arxiv_fail_count > 1:
                continue
            res.append(curr)
            if len(res) > 2000:
                break
    with open("../local_data/sample_1k_eval_data_0225.json", "w") as fo:
        fo.write(json.dumps(res, indent=4))
    with open("../local_data/sample_1k_eval_data_0225_test.json", "w") as fo:
        fo.write(json.dumps(res[:20], indent=4))

main()




import json
import os


def main():
    # input_dir = "/data/yubowang/ScholarCopilotLocal/evaluation/qwen_72b_output"
    # output_path = "../data/qwen_72b_eval_result_0303.json"
    # input_dir = "/data/yubowang/ScholarCopilotLocal/evaluation/sc_ul_output"
    # output_path = "../data/sc_ul_eval_result_0303.json"
    # input_dir = "/data/yubowang/ScholarCopilotLocal/evaluation/sc_2k_output"
    # output_path = "../data/sc_2k_eval_result_0303.json"
    input_dir = "/data/yubowang/ScholarCopilotLocal/evaluation/qwen_72b_re_output/"
    output_path = "../data/qwen_72b_re_eval_result_0314.json"
    # input_dir = "/data/yubowang/ScholarCopilotLocal/evaluation/qwen_7b_re_output/"
    # output_path = "../data/qwen_7b_re_eval_result_0314.json"
    input_data = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        with open(file_path, "r") as fi:
            curr_data = json.load(fi)
        input_data += curr_data
    input_data = sorted(input_data, key=lambda x: x["paper_id"])
    output_data = []
    for each in input_data:
        curr = {"paper_id": each["paper_id"],
                "title": each["title"],
                "abstract": each["abstract"],
                "paper": each["paper"],
                # "generated_text": each.get("qwen_2.5_72b_instruct_output", each.get("sc_generated_text", None)),
                "generated_text": each.get("qwen_2.5_7b_instruct_output",
                                           each.get("sc_generated_text", each.get("model_output", None))),
                "gpt4o_output": each["model_output"],
                "score": each["score"],
                "cost": each["cost"]}
        output_data.append(curr)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))
    sta_map = {}
    count = 0
    keys = ["Relevance", "Coherence", "Academic", "Completeness", "Innovation", "Total"]
    for each in input_data:
        if "score" not in each:
            print("score not in item", each)
            continue
        count += 1
        score_map = each["score"]
        flag = 0
        for k, v in score_map.items():
            for each_key in keys:
                if each_key in k:
                    flag += 1
                    if each_key not in sta_map:
                        sta_map[each_key] = 0.0
                    else:
                        sta_map[each_key] += float(v)
        if flag != 6:
            print("score_map failed", score_map)
    for k, v in sta_map.items():
        print("statistic", k, count, v / count)


main()





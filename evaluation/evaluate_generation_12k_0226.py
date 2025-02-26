import json
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_demo.scholar_copilot_model import *
from evaluate_retrieval_0224 import load_sc_model, load_eval_data
from tqdm import tqdm


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def sc_generate(model_info, text):
    model, tokenizer, device, meta_data, citation_map_data, index, lookup_indices = model_info
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_generate_full(model, tokenizer, device, current_text, 12000)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    # for each in yield_list:
    #     if "." in each and (each.endswith(".") or ".\n" in each):
    #         sentence_num += 1
    #         print("sentence_num: ", sentence_num, "each", each)
    #     curr_yield_text += " " + each
    # curr_prefix_length = len(curr_yield_text)
    while cite_start_hidden_state is not None and not enough:
        if cite_start_hidden_state == "<|continue|>":
            current_text = current_text
        else:
            retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
            reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
            reference_id_list.append(curr_index)
            current_text = current_text + reference
        current_text, cite_start_hidden_state = single_generate_full(model, tokenizer, device, current_text, 12000)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        # for each in yield_list:
        #     if "." in each and (each.endswith(".") or ".\n" in each):
        #         sentence_num += 1
        #         print("sentence_num: ", sentence_num, "each", each)
        #     curr_yield_text += " " + each
        # curr_prefix_length = len(curr_yield_text)
    display_text, post_citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    return display_text, citation_data_list


def format_input_text(item):
    title = item["title"]
    abstract = item["abstract"].replace("<|reference_start|>", "").replace("<|reference_end|>", "")
    format_text = f"Title:\n{title}\n\nAbstract:\n{abstract}\n\nIntroduction\n"
    return format_text


def load_exist_res(output_path):
    exist_ids, res = [], []
    if not os.path.exists(output_path):
        return exist_ids, res
    with open(output_path, "r") as fi:
        res = json.load(fi)
    for each in res:
        if "sc_generated_text" in each:
            exist_ids.append(each["paper_id"])
    return exist_ids, res


def eval_sc_generate():
    # output_path = "../data/eval_generation_result_0226_8k.json"
    output_path = "../data/eval_generation_result_0226_12k.json"
    model_info = load_sc_model(device="5")
    eval_data = load_eval_data()
    exist_ids, res = load_exist_res(output_path)
    for each in tqdm(eval_data):
        if each["paper_id"] in exist_ids:
            print("skipping", each["paper_id"])
            continue
        input_text = format_input_text(each)
        result_text, citation_data_list = sc_generate(model_info, input_text)
        each["citation_data_list"] = citation_data_list
        each["sc_generated_text"] = result_text
        res.append(each)
        with open(output_path, "w") as fo:
            fo.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    eval_sc_generate()



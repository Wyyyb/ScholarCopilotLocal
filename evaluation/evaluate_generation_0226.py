import json
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_demo.scholar_copilot_model import *
from evaluate_retrieval_0224 import *
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
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    for each in yield_list:
        if "." in each and (each.endswith(".") or ".\n" in each):
            sentence_num += 1
            print("sentence_num: ", sentence_num, "each", each)
        curr_yield_text += " " + each
    curr_prefix_length = len(curr_yield_text)
    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        for each in yield_list:
            if "." in each and (each.endswith(".") or ".\n" in each):
                sentence_num += 1
                print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
        curr_prefix_length = len(curr_yield_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    return display_text


def format_input_text(item):
    title = item["title"]
    abstract = item["abstract"].replace("<|reference_start|>", "").replace("<|reference_end|>", "")
    format_text = f"Title:\n{title}\n\nAbstract:\n{abstract}\n\nIntroduction:\n"
    return format_text


def eval_sc_generate():
    output_path = "../data/eval_generation_result_0226.json"
    model_info = load_sc_model()
    eval_data = load_eval_data()
    res = []
    for each in tqdm(eval_data):
        input_text = format_input_text(each)
        result_text = sc_generate(model_info, input_text)
        each["sc_generated_text"] = result_text
        res.append(each)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=4))


main()


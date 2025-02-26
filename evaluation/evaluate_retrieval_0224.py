import json
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_demo.scholar_copilot_model import *
from tqdm import tqdm


def load_eval_data():
    with open("../data/eval_re_data_1k_0225.json", "r") as fi:
        eval_data = json.load(fi)
    return eval_data


def format_paper(eval_item):
    title = eval_item["title"]
    abstract = eval_item["abstract"]
    paper = eval_item["paper"]
    return f"Title:\n{title}\n\nAbstract:\n{abstract}\n\n{paper}"


def format_reference(abstract):
    return f"<|cite_start|> (Reference: {abstract}) <|cite_end|>"


def single_eval_sc(model_info, eval_item, top_k=10):
    bib_info = eval_item["bib_info"]
    paper_text = format_paper(eval_item)
    eval_score = []
    for k, v in bib_info.items():
        gt = []
        for each in v:
            if each["citation_corpus_id"].startswith("ss"):
                print("gt not in meta data", each["citation_corpus_id"])
            gt.append(each["citation_corpus_id"])
        # print("gt", gt)
        start_index = paper_text.index(k)
        input_text = paper_text[:start_index]
        citations = generate_citation(model_info, input_text, top_k=top_k)
        curr_eval_score = {}
        for tpk in range(top_k):
            retrieval_score = False
            for each in citations[:tpk + 1]:
                if each in gt:
                    retrieval_score = True
            curr_eval_score[f"top_{str(tpk+1)}_score"] = retrieval_score
        eval_score.append(curr_eval_score)
    eval_item["eval_score"] = eval_score
    statistic = {}
    for tpk in range(top_k):
        right_count = 0.0
        wrong_count = 0.0
        for each in eval_score:
            if each[f"top_{str(tpk+1)}_score"] is True:
                right_count += 1
            else:
                wrong_count += 1
        accu = right_count / (right_count + wrong_count)
        statistic[f"top_{str(tpk+1)}_score"] = {"right_count": right_count,
                                                "wrong_count": wrong_count,
                                                "accu": accu}
    eval_item["statistic"] = statistic
    print("single statistic", statistic)
    return eval_item


def generate_citation(model_info, input_text, top_k):
    model, tokenizer, device, meta_data, citation_map_data, index, lookup_indices = model_info
    new_input_text = input_text + " <|cite_start|>"
    new_input = tokenizer(new_input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        new_output = model(
            new_input.input_ids,
            attention_mask=new_input.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    cite_rep = new_output.hidden_states[-1][:, -1, :]
    retrieved_k_results = retrieve_reference(index, lookup_indices, cite_rep, top_k=top_k)
    searched_citations = []
    for each in retrieved_k_results:
        curr_index, distance = each
        # print("index", curr_index)
        if curr_index not in meta_data:
            print("index not found in meta_data", curr_index)
            continue
        paper_id = meta_data[curr_index]["paper_id"]
        # print("paper_id", paper_id)
        citation_info = citation_map_data[paper_id]
        # print("generate_citation citation_info", citation_info)
        # searched_citations.append(citation_info)
        searched_citations.append(curr_index)
    return searched_citations


def compute_overall(eval_res, top_k=10):
    overall_res = {}
    for tpk in range(top_k):
        right_count = 0.0
        wrong_count = 0.0
        for item in eval_res:
            statistic = item["statistic"]
            curr = statistic[f"top_{str(tpk+1)}_score"]
            right_count += curr["right_count"]
            wrong_count += curr["wrong_count"]
        accu = right_count / (right_count + wrong_count)
        overall_res[f"top_{str(tpk+1)}_score"] = {"right_count": right_count,
                                                  "wrong_count": wrong_count,
                                                  "accu": accu}
    return overall_res


def eval_sc_retrieval():
    output_path = "../data/eval_retrieve_result_0226.json"
    model_info = load_sc_model()
    eval_data = load_eval_data()
    res = []
    for each in tqdm(eval_data):
        start = time.time()
        single_res = single_eval_sc(model_info, each)
        print("single costing time:", time.time() - start)
        res.append(single_res)
    overall_res = compute_overall(res)
    print("overall_res", overall_res)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=4))
    with open("../data/result_summary_0226.json", "w") as fo:
        fo.write(json.dumps(overall_res, indent=4))


def load_sc_model():
    model_path = "../model_v1208/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)
    meta_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    meta_data = load_meta_data(meta_data_path)
    citation_map_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)
    index_dir = "../data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")
    return model, tokenizer, device, meta_data, citation_map_data, index, lookup_indices


if __name__ == "__main__":
    print("evaluating scholar copilot retrieval")
    eval_sc_retrieval()


import json
import time
import torch
import os
import sys
import spacy
import re

from pyserini.search.lucene import LuceneSearcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def load_eval_data():
    with open("./eval_re_data_1k_0225.json", "r") as fi:
        eval_data = json.load(fi)
    return eval_data


def format_paper(eval_item):
    title = eval_item["title"]
    abstract = eval_item["abstract"]
    paper = eval_item["paper"]
    return f"Title:\n{title}\n\nAbstract:\n{abstract}\n\n{paper}"


def single_eval_sc(searcher, eval_item, top_k=10):
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
        text = paper_text[:start_index]
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        input_text = sentences[-1].split('|>')[-1].strip().split('\n')[-1]
        # if len(input_text) < 10:
            # input_text = text[-10:]
        print("input_text", input_text)
        citations = searcher.search(input_text, k=top_k)
        curr_eval_score = {}
        for tpk in range(top_k):
            retrieval_score = False
            for each in citations[:tpk + 1]:
                if each.docid in gt:
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
    eval_data = load_eval_data()
    searcher = LuceneSearcher('./index')
    res = []
    for each in tqdm(eval_data):
        start = time.time()
        single_res = single_eval_sc(searcher, each)
        print("single costing time:", time.time() - start)
        res.append(single_res)
    overall_res = compute_overall(res)
    print("overall_res", overall_res)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=4))
    with open("../data/result_summary_0226.json", "w") as fo:
        fo.write(json.dumps(overall_res, indent=4))

if __name__ == "__main__":
    print("evaluating scholar copilot retrieval")
    eval_sc_retrieval()


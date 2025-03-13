from vllm import LLM, SamplingParams
from typing import List
import json
import os
import sys
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import glob
from tqdm import tqdm
from itertools import chain
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tevatron_retrieval.search_mistral_e5 import configure_faiss_for_gpu, load_index_and_data
from tevatron_retrieval.search_mistral_e5 import get_query_embedding, get_detailed_instruct


def load_vllm_model(model_path: str):
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|citation|>"]
        # 初始化模型
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))  # 根据GPU数量调整
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_path.lower()
                else None
            ),
            top_p=1.0
        )
        return llm, sampling_params
    except Exception as e:
        print("load vllm model failed", e)
        return None, None


def batch_predict(llm, sampling_params, prompts: List[str]) -> List[str]:
    if not llm or not sampling_params:
        print("llm, sampling_params are None")
        return []
    try:
        print("Processing", len(prompts))
        outputs = llm.generate(prompts, sampling_params)
        # 提取生成的文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        print("Generated", results)
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []


def format_prompt(title, abstract, existing_part):
    initial_prompt = "You are a PhD student in Computer Science. I will provide you " \
                     "with the title, abstract, and a portion of a computer science paper. " \
                     "Please help me complete the introduction and related work sections of " \
                     "this paper. For any citations needed, please use <|citation|> as a " \
                     "placeholder.Please pause when you generate a <|citation|> so I can " \
                     "provide the actual citation. When you feel your task is complete, " \
                     "please indicate this by generating <|end_section|>."

    # return f"{initial_prompt}\nTitle: {title}\nAbstract: {abstract}\n{existing_part}"
    return f"{initial_prompt}\n{existing_part}"


def load_retriever():
    index_files = glob.glob("/home/xueguangma/arxiv-llm/tevatron_retrieval/corpus*.pkl")  # Replace with actual path
    retriever, look_up = load_index_and_data(index_files)
    retriever = configure_faiss_for_gpu(retriever)

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
    return retriever, look_up, model, tokenizer


def load_corpus_data(corpus_data_path):
    print("loading corpus data...")
    meta_data = {}
    with open(corpus_data_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            if curr["corpus_id"] not in meta_data:
                meta_data[curr["corpus_id"]] = curr
    print("corpus data loaded.")
    return meta_data
        

def single_retrieve(retriever, look_up, model, tokenizer, query):
    task = 'Given a paper passage, retrieve the most proper paper to cite next.'
    query = get_detailed_instruct(task, query)
    query_embedding = get_query_embedding(model, tokenizer, query)

    # Perform Search
    k = 1
    scores, indices = retriever.search(query_embedding, k)
    documents = [look_up[i] for i in indices[0]]
    # Output Results
    print("Scores:", scores)
    # print("Indices:", indices)
    # print("Retrieved Documents:", [look_up[i] for i in indices[0]])
    return documents[0]


def find_last_complete_sentence(text):
    last_period = text.rfind('.')

    if last_period == -1:
        return ""

    result = text[last_period + 1:].strip()

    if len(result) < 5:
        second_last_period = text.rfind('.', 0, last_period)
        if second_last_period != -1:
            result = text[second_last_period + 1:].strip()

    return result


def single_complete(generation_model, retrieval_model, corpus_data, title, abstract, existing_content):
    prompt = format_prompt(title, abstract, existing_content)
    llm, sampling_params = generation_model
    output_text = batch_predict(llm, sampling_params, [prompt])
    output_text = output_text[0]
    start_index = output_text.find("<|citation|>")
    curr_text = output_text[:start_index]
    last_sen = find_last_complete_sentence(curr_text)

    retriever, look_up, model, tokenizer = retrieval_model
    retrieved_id = single_retrieve(retriever, look_up, model, tokenizer, last_sen)
    cite_key = corpus_data[retrieved_id]["citation_key"]
    generated_text = curr_text + "~\\cite{" + cite_key + "} "
    return generated_text


def single_item_eval(generation_model, retrieval_model, corpus_data, item):
    title = item["title"]
    abstract = item["abstract"].replace("<|reference_start|>", "").replace("<|reference_end|>", "")
    existing_content = "\nIntroduction\n"
    output_text = single_complete(generation_model, retrieval_model, corpus_data, title, abstract, existing_content)
    while len(output_text) < 10000 and "<|end_section|>" not in output_text:
        output_text = single_complete(generation_model, retrieval_model, corpus_data, title, abstract, output_text)
    return output_text


def load_eval_data():
    with open("../data/eval_re_data_1k_0225.json", "r") as fi:
        eval_data = json.load(fi)
    return eval_data


def eval_qwen_generation(model_path):
    output_path = "../data/qwen_7b_eval_generation_result_re_0313.json"
    eval_data = load_eval_data()
    eval_data = eval_data[:10]
    llm, sampling_params = load_vllm_model(model_path)
    generation_model = (llm, sampling_params)

    retriever, look_up, model, tokenizer = load_retriever()
    retrieval_model = (retriever, look_up, model, tokenizer)

    corpus_data = load_corpus_data("../local_data/corpus_data_arxiv_1215.jsonl")

    res = []
    for i, each in tqdm(enumerate(eval_data)):
        output_text = single_item_eval(generation_model, retrieval_model, corpus_data, each)
        eval_data[i]["qwen_2.5_72b_instruct_output"] = output_text
        res.append(eval_data[i])

    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    eval_qwen_generation("/data/yubowang/models/Qwen2.5-7B-Instruct")

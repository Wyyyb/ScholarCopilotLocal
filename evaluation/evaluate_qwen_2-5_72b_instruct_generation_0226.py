from vllm import LLM, SamplingParams
from typing import List
import json
import os
import random

random.seed(12345)


def load_vllm_model(model_path: str):
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
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
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []


def load_eval_data():
    with open("../data/eval_re_data_1k_0225.json", "r") as fi:
        eval_data = json.load(fi)
    return eval_data


def format_prompt(title, abstract, reference_list):
    initial_prompt = "You are a PhD student in Computer Science." \
                     "I will provide you with the title and abstract of a computer science paper, " \
                     "along with some related references you might need. Please complete the introduction and " \
                     "related work sections of this paper."
    reference = ""
    for each in reference_list:
        citation_key = each[0]
        reference = each[1]
        reference += f"reference key: {citation_key}\nreference content: {reference}\n\n"
    return f"{initial_prompt}\nTitle: {title}\nAbstract: {abstract}\nReferences:\n{reference}\nIntroduction\n"


def eval_qwen_generation(model_path):
    output_path = "../data/qwen_eval_generation_result_0228.json"
    llm, sampling_params = load_vllm_model(model_path)
    eval_data = load_eval_data()
    eval_data = eval_data[:10]
    prompts = []
    for each in eval_data:
        title = each["title"]
        abstract = each["abstract"]
        reference_list = []
        for item in each["bib_info"]:
            for each_item in item:
                reference_list.append([each_item["citation_key"], each_item["abstract"]])
        random.shuffle(reference_list)
        prompts.append(format_prompt(title, abstract, reference_list))
    model_outputs = batch_predict(llm, sampling_params, prompts)
    res = []
    if len(model_outputs) != len(eval_data):
        print("inconsistent model output number", len(model_outputs), len(eval_data))
    for i in range(len(model_outputs)):
        eval_data[i]["qwen_2.5_72b_instruct_output"] = model_outputs[i]
        res.append(eval_data[i])
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    eval_qwen_generation("../models/Qwen2.5-72B-Instruct")

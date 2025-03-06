import json
import os
from multiprocessing import Process
import time
from tqdm import tqdm
from typing import Callable
from openai import OpenAI
from pathlib import Path
import requests


def process_chunk(start_idx: int,
                  end_idx: int,
                  input_path: str,
                  output_dir: str,
                  prompt_func: Callable,
                  process_id: int):
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{start_idx + 1}-{end_idx}.json')

    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = {item['paper_id']: item for item in json.load(f)}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_data = data[start_idx:end_idx]

    client = OpenAI()

    results = []

    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        if item['paper_id'] in existing_results:
            results.append(existing_results[item['paper_id']])
            continue

        try:
            messages = prompt_func(item)

            # 调用GPT-4
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.3,
                max_tokens=3200,
                top_p=0.95
            )
            item['model_output'] = completion.choices[0].message.content
            item["score"] = extract_scores(item['model_output'])
            if item["score"] is None:
                print("extracting scores failed")
                continue
            item['cost'] = completion.usage.completion_tokens * 10 / 1e6 + completion.usage.prompt_tokens * 2.5 / 1e6
            results.append(item)

            if len(results) % 2 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['paper_id']}: {str(e)}")
            continue
            # item['error'] = str(e)
            # results.append(item)

        time.sleep(0.1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(input_path: str,
                          output_dir: str,
                          prompt_func: Callable,
                          num_processes: int = 1):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # for test
        # data = data[:50]
        total_items = len(data)
    print("len(data)", len(data))
    # add idx
    for i, each in enumerate(data):
        if "paper_id" not in each:
            data[i]["paper_id"] = i
    # with open(input_path, "w") as fo:
    #     fo.write(json.dumps(data, indent=4))

    chunk_size = total_items // num_processes

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else total_items

        p = Process(
            target=process_chunk,
            args=(start_idx, end_idx, input_path, output_dir, prompt_func, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed!")


def example_prompt_func(item):
    title = item["title"]
    abstract = item["abstract"].replace("<|reference_start|>", "").replace("<|reference_end|>", "")
    ground_truth = item["paper"]
    generated_text = item.get("sc_generated_text", item.get("qwen_2.5_7b_instruct_output", None))
    if not generated_text:
        print("generated_text is empty")
    prompt = generate_evaluation_prompt(title, abstract, generated_text, ground_truth)
    chat_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        },
    ]
    return chat_prompt


def generate_evaluation_prompt(title: str, abstract: str, generated_text: str, ground_truth: str) -> str:
    prompt = """You are a senior computer science scholar. Please evaluate the AI-generated content using the ground truth as reference.

Evaluate the following five dimensions by comparing the AI-generated content with the ground truth:

[Detailed Evaluation]
1. Content Relevance:
- Key strengths:
- Main gaps:
- Comparison with ground truth:

2. Logical Coherence:
- Key strengths:
- Main gaps:
- Comparison with ground truth:

3. Academic Standards:
- Key strengths:
- Main gaps:
- Comparison with ground truth:

4. Background Completeness:
- Key strengths:
- Main gaps:
- Comparison with ground truth:

5. Innovation Statement:
- Key strengths:
- Main gaps:
- Comparison with ground truth:
[End Evaluation]

[Improvement Suggestions]
1.
2.
3.
[End Suggestions]

Based on your above analysis, provide numerical scores in the following format:
[Scores]
Relevance: <score>/5
Coherence: <score>/5
Academic: <score>/5
Completeness: <score>/5
Innovation: <score>/5
Total: <sum>/25
[End Scores]

Below are the materials for evaluation:

Paper Title:
{title}

Abstract:
{abstract}

Ground Truth Content:
{ground_truth}

AI Generated Content:
{generated_text}

Remember to first provide detailed evaluation, then improvement suggestions, and finally the numerical scores in the exact format specified above."""

    return prompt.format(
        title=title,
        abstract=abstract,
        ground_truth=ground_truth,
        generated_text=generated_text
    )


def extract_scores(gpt_response: str):
    """
    Extract scores from GPT's response.

    Args:
        gpt_response (str): The full response from GPT

    Returns:
        dict: Dictionary containing the scores
    """
    try:
        # Extract the scores section
        if "[Scores]" not in gpt_response:
            gpt_response = gpt_response.replace("### Scores", "[Scores]")
            print("found another [Scores]")
        scores_section = gpt_response.split("[Scores]")[1].split("[End Scores]")[0].strip()

        # Parse individual scores
        scores = {}
        for line in scores_section.split('\n'):
            if ':' in line:
                key, value = line.split(':')
                # Extract the numeric score before the '/5'
                try:
                    score = float(value.strip().split('/')[0])
                except Exception as e:
                    print("Error parsing", line, e)
                    return None
                scores[key.strip()] = score

        return scores
    except Exception as e:
        print(f"Error parsing scores: {e}")
        return None


if __name__ == "__main__":
    # input_path = "../data/eval_generation_result_unlimit_0228.json"
    # output_dir = "sc_ul_output/"
    input_path = "../data/qwen_7b_gt_eval_generation_result_0305.json"
    output_dir = "qwen_7b_gt_output/"
    os.makedirs(output_dir, exist_ok=True)
    process_large_dataset(
        input_path=input_path,
        output_dir=output_dir,
        prompt_func=example_prompt_func,
        num_processes=50
    )



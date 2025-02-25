import random
import json


def single_process(item):
    ori_intro = item["full_intro"]
    paper_id = item["paper_id"]
    abstract = item["abstract"]
    title = item["title"]
    ori_bib_info = item["bib_info"]
    output_text, bib_info = process_citations(ori_intro, ori_bib_info)
    res = {"paper_id": paper_id, "title": title, "abstract": abstract, "paper": output_text,
           "bib_info": bib_info}
    return res


def process_citations(input_text, bib_info):
    import re

    # 第一步：处理citations，但保持原有编号
    temp_bib_info = {}
    result_text = input_text

    # 找出所有citations和它们的位置
    citations = []
    for match in re.finditer(r'<\|(?:multi_)?cite_[0-9_]+\|>', input_text):
        citations.append(match.group())

    i = 0
    while i < len(citations):
        cite = citations[i]

        # 处理multi_cite
        if 'multi_cite' in cite:
            base_num = re.search(r'multi_cite_(\d+)', cite).group(1)

            # 收集所有连续的相同base number的multi_cite
            multi_cites = []
            while i < len(citations) and f'multi_cite_{base_num}' in citations[i]:
                current_cite = citations[i]
                # 检查当前multi_cite是否在bib_info中
                if current_cite in bib_info:
                    multi_cites.append(current_cite)
                else:
                    # 如果不在bib_info中，删除这个citation
                    result_text = result_text.replace(current_cite, '', 1)
                i += 1

            # 如果没有有效的multi_cite，继续下一个处理
            if not multi_cites:
                continue

            # 只保留第一个multi_cite
            first_cite = multi_cites[0]
            temp_bib_info[first_cite] = [bib_info[mc] for mc in multi_cites]

            # 删除除第一个外的所有multi_cite
            for mc in multi_cites[1:]:
                result_text = result_text.replace(mc, '', 1)

        # 处理普通cite
        else:
            if cite not in bib_info:
                result_text = result_text.replace(cite, '', 1)
            else:
                temp_bib_info[cite] = [bib_info[cite]]
            i += 1

    # 第二步：重新编号
    # 再次找出所有剩余的citations（已经处理过multi_cite）
    remaining_citations = []
    for match in re.finditer(r'<\|(?:multi_)?cite_[0-9_]+\|>', result_text):
        remaining_citations.append((match.group(), match.start()))

    # 按位置排序
    remaining_citations.sort(key=lambda x: x[1])

    # 创建新的编号映射
    cite_mapping = {}
    new_bib_info = {}
    for i, (old_cite, _) in enumerate(remaining_citations):
        new_cite = f'<|cite_{i}|>'
        cite_mapping[old_cite] = new_cite
        new_bib_info[new_cite] = temp_bib_info[old_cite]

    # 按照位置从后向前替换，以避免位置变化影响
    final_text = result_text
    for old_cite, pos in reversed(remaining_citations):
        new_cite = cite_mapping[old_cite]
        final_text = final_text[:pos] + new_cite + final_text[pos + len(old_cite):]

    return final_text, new_bib_info


def process_citations_bk(input_text, bib_info):
    import re

    # 第一步：处理citations，但保持原有编号
    temp_bib_info = {}
    result_text = input_text

    # 找出所有citations和它们的位置
    citations = []
    for match in re.finditer(r'<\|(?:multi_)?cite_[0-9_]+\|>', input_text):
        citations.append(match.group())

    i = 0
    while i < len(citations):
        cite = citations[i]
        if cite == "<|multi_cite_15_4|>":
            cite = cite
            s = 1
        # 如果citation不在输入的bib_info中,跳过
        if cite not in bib_info:
            result_text = result_text.replace(cite, '', 1)
            i += 1
            continue

        # 处理multi_cite
        if 'multi_cite' in cite:
            base_num = re.search(r'multi_cite_(\d+)', cite).group(1)

            # 收集所有连续的相同base number的multi_cite
            multi_cites = []
            while i < len(citations) and f'multi_cite_{base_num}' in citations[i]:
                multi_cites.append(citations[i])
                i += 1

            # 只保留第一个multi_cite
            first_cite = multi_cites[0]
            print("bib_info", bib_info)
            temp_bib_info[first_cite] = [bib_info[mc] for mc in multi_cites]

            # 删除除第一个外的所有multi_cite
            for mc in multi_cites[1:]:
                result_text = result_text.replace(mc, '', 1)

        # 处理普通cite
        else:
            temp_bib_info[cite] = [bib_info[cite]]
            i += 1

    # 第二步：重新编号
    # 再次找出所有剩余的citations（已经处理过multi_cite）
    remaining_citations = []
    for match in re.finditer(r'<\|(?:multi_)?cite_[0-9_]+\|>', result_text):
        remaining_citations.append((match.group(), match.start()))

    # 按位置排序
    remaining_citations.sort(key=lambda x: x[1])

    # 创建新的编号映射
    cite_mapping = {}
    new_bib_info = {}
    for i, (old_cite, _) in enumerate(remaining_citations):
        new_cite = f'<|cite_{i}|>'
        cite_mapping[old_cite] = new_cite
        new_bib_info[new_cite] = temp_bib_info[old_cite]

    # 按照位置从后向前替换，以避免位置变化影响
    final_text = result_text
    for old_cite, pos in reversed(remaining_citations):
        new_cite = cite_mapping[old_cite]
        final_text = final_text[:pos] + new_cite + final_text[pos + len(old_cite):]

    return final_text, new_bib_info


def main():
    output_path = "../local_data/eval_re_data_1k_0225.json"
    with open("../local_data/sample_1k_eval_data_0225.json", "r") as fi:
        data = json.load(fi)
    res_data = []
    for each in data:
        single_res = single_process(each)
        res_data.append(single_res)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res_data, indent=4))


def test():
    # 测试数据
    # input_text = "XYZ<|cite_3|>xyz<|multi_cite_4_1|><|multi_cite_4_2|>XYZ<|cite_1|>" \
    #              "kajnfjksaf<|cite_2|>sss<|cite_5|>safnjn<|multi_cite_6_1|><|multi_cite_6_2|>"
    # bib_info = {
    #     "<|cite_1|>": "XYZ",
    #     "<|cite_2|>": "PQR",
    #     "<|cite_3|>": "cite_3",
    #     "<|multi_cite_4_1|>": "ABC",
    #     "<|multi_cite_4_2|>": "JQK",
    #     "<|cite_5|>": "cite_5",
    #     "<|multi_cite_6_1|>": "multi_cite_6_1",
    #     "<|multi_cite_6_2|>": "multi_cite_6_2",
    # }
    input_text = "<|cite_4|>X YZ<|cite_2|> XYZ<|multi_cite_4_1|><|multi_cite_4_2|>XYZ <|cite_1|>"
    bib_info = {
        "<|cite_1|>": "XYZ",
        "<|cite_2|>": "PQR",
        "<|multi_cite_4_1|>": "ABC",
        "<|multi_cite_4_2|>": "JQK"
    }

    # 运行函数
    output_text, new_bib_info = process_citations(input_text, bib_info)

    # 打印结果
    print("Output text:", output_text)
    print("New bib info:", new_bib_info)


main()


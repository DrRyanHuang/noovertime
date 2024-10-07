import re
import json
from prompts import INTENT_DISASSEMBLE_PROMPT
from config import AISTUDIO_AK

import erniebot
from tqdm import tqdm
from utils import pp_print


def parse_text(text):

    # 使用正则表达式匹配所有[]中的内容
    # 这里的正则表达式解释如下：
    # \[ 匹配左方括号 [
    # (.*?) 匹配任意字符（非贪婪模式），尽可能少的匹配字符
    # \] 匹配右方括号 ]
    matches = re.findall(r"\[(.*?)\]", text)

    # 输出匹配结果
    # print(matches)
    return matches


def query_split(query, model="ernie-3.5"):

    content = INTENT_DISASSEMBLE_PROMPT["content"].replace("{input_query}", query)

    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )

    pp_print("[子问题拆分合并]", response.get_result())
    response = response.get_result()

    return parse_text(response)


def query_split_need_tools(query, need_description_list, model="ernie-3.5"):

    need_description_list_str = "\n" + "\n".join(
        [f"{_idx}. {desc}" for _idx, desc in enumerate(need_description_list, 1)]
    )

    content = (
        INTENT_DISASSEMBLE_PROMPT["content"]
        .replace("{input_query}", query)
        .replace("{need_description_list}", need_description_list_str)
    )

    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )

    pp_print("[子问题拆分列表]", response.get_result())
    response = response.get_result()

    return parse_text(response)


if __name__ == "__main__":

    erniebot.api_type = "aistudio"
    erniebot.access_token = AISTUDIO_AK

    text = "<拆分问题列表>\n[今年10月7号从深圳回杭州的机票最便宜是多少钱？]\n[今年10月7号深圳的最高温度是多少？]"
    matches = parse_text(text)
    print(matches)

    # query = "今年10月7号从深圳回杭州机票最便宜多少钱？当天在深圳最高温度是多少？"
    # query_list = query_split(query)

    with open("dataset.json", "r") as f:
        query_info_list = json.load(f)

    query_info_new_list = []
    for query_info in tqdm(query_info_list):
        # print(query_info)
        query_split_list = query_split(query_info["query"])
        query_info_new = query_info.copy()
        query_info_new["split"] = query_split_list
        query_info_new_list.append(query_info_new)

    json.dump(
        query_info_new_list,
        open("dataset_split.json", "w"),
        ensure_ascii=False,
        indent=4,
    )

import requests
import os
import json
import argparse
import qianfan
import erniebot
from utils import (
    function_request_yiyan_qianfan,
    function_request_yiyan_aistudio,
    request_plugin,
)

import time
from const import (
    SIGNIFICANT_MAPPINGS,
    TOOL2ID,
    TERM_DATE_LUNAR_DATE,
    find_solar_terms,
    hit_keywords,
)
from config import QIANFAN_AK, QIANFAN_SK, AISTUDIO_AK
from pprint import pprint
from utils import (
    run_once,
    api_list_process,
    pp_print,
    find_and_convert_date,
    convert_to_lunar_date2str,
    get_weekday,
    get_tools_description,
)
from query_split import query_split, query_split_need_tools
from answer_combine import answer_combine
from prompts import ADJUST_ANSWER_PROMPT
from retrieve import get_topk


ERNIE_MODEL = "ernie-4.0-turbo-8k"

erniebot.api_type = "aistudio"
erniebot.access_token = AISTUDIO_AK

# List supported models
models = erniebot.Model.list()
# pprint(models)


def retrieve_api(query, k, description_list):
    """
    根据给定的查询语句和API文件路径, 从中检索与查询语句最相关的k个API。

    Args:
        query (str): 查询语句。
        api_path (str): API文件的路径。
        k (int): 返回的召回的API数量。

    Returns:
        list: 检索到的k个API的json格式描述信息列表。

    """

    topk_api_id = get_topk(query, description_list, k)

    return topk_api_id


def tool_use_aistudio2queryanalysis(query, query_id, api_path, save_path, topK):
    """
    通过调用API获取相关函数, 并使用聊天补全模型生成回答，加入query拆分 + 回答合并

    Args:
        query: 用户输入的查询语句
        api_path: API的查询路径
        save_path: 保存答案的文件路径
        topK: 返回的召回的API数量

    Returns:
        None

    """
    # 所有工具
    description_list, description_dict = get_tools_description(api_path)

    # 关键词命中优先级最高
    keywords_hit = hit_keywords(query)
    # 做召回
    retrieve_list = retrieve_api(query, topK, description_list)

    topk_api_id = set(keywords_hit) | set(retrieve_list)
    retrieve_list = [description_dict[description_list[id]] for id in topk_api_id]

    need_description_list = [tool["description"] for tool in retrieve_list]

    # 先做query分析, 把要传入的工具描述也传入
    # ---------------------------------------------------------------------------------------

    sub_query_list = query_split_need_tools(query, need_description_list)
    description_list, description_dict = get_tools_description(api_path)

    relevant_APIs = []
    answer = {"query": query, "query_id": query_id}
    pp_print(f"[用户query]：{query}")

    sub_query_answer = [""] * len(sub_query_list)

    for idx, sub_query in enumerate(sub_query_list):

        # 关键词命中优先级最高
        keywords_hit = hit_keywords(sub_query)
        # 做召回
        retrieve_list = retrieve_api(sub_query, topK, description_list)

        topk_api_id = set(keywords_hit) | set(retrieve_list)

        retrieve_list = [description_dict[description_list[id]] for id in topk_api_id]

        # 对API列表进行处理, 获取url路径列表和标准API信息列表
        paths_list, api_list = api_list_process(retrieve_list)

        # -------- basic 形式（子query知识补充） --------
        sth_need = ""

        if "限行" in sub_query:
            date_str = find_and_convert_date(sub_query)
            date_weekday_str = ""
            if len(date_str):
                weekday = get_weekday(*[int(x) for x in date_str.split("-")])
                date_weekday_str = f"注意: {date_str}是{weekday}"

            other_std_need = ""
            if "北京" in sub_query:
                # https://bj.bendibao.com/news/xianxingchaxun/
                other_std_need = (
                    "注意：北京在2024年9月30日至2024年12月29日期间，星期一限行尾号为3和8，星期二限行尾号为4和9，"
                    "星期三限行尾号为5和0，星期四限行尾号为1和6，星期五限行尾号为2和7，周六日和假期不限行"
                )

            sth_need = (
                "注意: 将北京、武汉和成都日期参数 date_or_day_of_week 解析为星期几的形式，如星期一，"
                "天津的日期参数 date_or_day_of_week 解析为日期，如2024年9月1日"
                "注意: city 参数必须解析为城市，如北京，上海等\n"
                "注意: bd_gov_xianxing 工具返回的内容为空时, 视为没有限行\n"
            )
            sth_need = f"{date_weekday_str}\n{sth_need}\n{other_std_need}"

        if "农历" in sub_query or "阴历" in sub_query:
            date_str = find_and_convert_date(sub_query)
            if len(date_str):
                lunar_data_str = convert_to_lunar_date2str(
                    *[int(x) for x in date_str.split("-")]
                )
                sth_need += f"注意: {date_str}是{lunar_data_str}"
            else:
                sth_need += ""

        # -------- 如果 query 中存在节气 --------
        terms = find_solar_terms(sub_query)
        if terms:
            for term in terms:
                sth_need += "\n" + TERM_DATE_LUNAR_DATE[term]

        messages = [
            {
                "role": "user",
                "content": f"{sub_query}\n{sth_need}",
            }
        ]
        # 超出10轮就退出
        n = 0
        while n < 10:
            # 请求一言模型失败也退出
            try:
                response, func_name, kwargs = function_request_yiyan_aistudio(
                    None, messages, api_list
                )  # 这里找一个最合适的API
            except Exception as e:
                pp_print(e)
                break

            if isinstance(response, str):  # response['function_call']
                pp_print(f"[智能体回答]：{response}")
                # answer["result"] = response  # <--------------- 只能从这里退出
                sub_query_answer[idx] = response
                break
            relevant_APIs.append({"api_name": func_name, "required_parameters": kwargs})
            pp_print(f"[调用函数]：{func_name}, 参数：{kwargs}")

            try:
                paths = next(
                    item["paths"] for item in paths_list if item["name"] == func_name
                )  # '/plugins?id=14'
            except StopIteration:
                pp_print("模型由于幻觉生成不存在的工具")
                continue

            func_response = request_plugin(paths, kwargs)
            func_content = json.dumps({"return": func_response}, ensure_ascii=False)
            pp_print(f"[函数返回]：{func_response}")

            assert hasattr(response, "function_call")
            function_call = response.function_call

            assert "thoughts" in response.get_result()
            pp_print("[Thought]:", response.get_result()["thoughts"])

            messages += [
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call,
                },
                {
                    "role": "function",
                    "name": function_call["name"],
                    "content": func_content,
                },
            ]
            n += 1

            # 防止请求一言频率过高, 休眠0.5s
            time.sleep(0.5)

    answer["result"] = answer_combine(
        query, sub_query_list, sub_query_answer, model=ERNIE_MODEL
    )
    answer["relevant APIs"] = relevant_APIs
    if not answer.get("result"):
        answer["result"] = "抱歉, 无法回答您的问题。"
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(answer, ensure_ascii=False) + "\n")


def start(test_path, api_path, save_path, topK=5):
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 测试集
        for __idx, line in enumerate(data):

            pp_print("--------------------------------")

            query = line["query"]
            query_id = line["qid"]

            # if "立秋一般是阴历什么时候。2024年立秋的" not in query:
            #     continue

            if __idx <= 12:
                continue

            # tool_use_qianfan(query, query_id, api_path, save_path, topK)
            # tool_use_aistudio(query, query_id, api_path, save_path, topK)
            # tool_use_aistudio_ntimes(query, query_id, api_path, save_path, topK)
            tool_use_aistudio2queryanalysis(query, query_id, api_path, save_path, topK)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--AK", type=str, default="", help="一言的api key")
    parser.add_argument("--SK", type=str, default="", help="一言的secret key")
    parser.add_argument(
        "--test_path", type=str, default="dataset.json", help="测试集的路径"
    )
    parser.add_argument(
        "--api_path",
        type=str,
        default="api_list.json",
        help="API集合的路径, 所有可用的API都在这个文件中",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./result.json",
        help="保存智能体回答结果的路径",
    )
    parser.add_argument(
        "--topK", type=int, default=5, help="在API检索阶段, 召回的API数量"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args()

    os.environ["QIANFAN_AK"] = QIANFAN_AK
    os.environ["QIANFAN_SK"] = QIANFAN_SK

    start(args.test_path, args.api_path, args.save_path, topK=5)

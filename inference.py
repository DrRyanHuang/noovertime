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
from retrieve import get_topk_baseline, get_topk
import time
from const import SIGNIFICANT_MAPPINGS, TOOL2ID
from config import QIANFAN_AK, QIANFAN_SK, AISTUDIO_AK
from pprint import pprint
from utils import run_once, api_list_process
from query_split import query_split
from answer_combine import answer_combine

erniebot.api_type = "aistudio"
erniebot.access_token = AISTUDIO_AK

# List supported models
models = erniebot.Model.list()
pprint(models)


@run_once
def get_tools_description(api_path):

    # 「描述-api名字」对组成的字典
    description_dict = {}
    with open(api_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            api_json = json.loads(line)
            description_dict[api_json["description"]] = api_json
        description_list = list(description_dict.keys())

    return description_list, description_dict


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


def hit_keywords(query):
    tools = []
    for keyword, __tools in SIGNIFICANT_MAPPINGS.items():
        if keyword in query:
            tools += __tools

    return [TOOL2ID[tool_str] for tool_str in tools]


def tool_use_qianfan(query, query_id, api_path, save_path, topK):
    """
    通过调用API获取相关函数, 并使用聊天补全模型生成回答

    Args:
        query: 用户输入的查询语句
        api_path: API的查询路径
        save_path: 保存答案的文件路径
        topK: 返回的召回的API数量

    Returns:
        None

    """
    description_list, description_dict = get_tools_description(api_path)

    # 关键词命中优先级最高
    keywords_hit = hit_keywords(query)
    # 做召回
    retrieve_list = retrieve_api(query, topK, description_list)

    topk_api_id = set(keywords_hit) | set(retrieve_list)
    retrieve_list = [description_dict[description_list[id]] for id in topk_api_id]

    # 对API列表进行处理, 获取url路径列表和标准API信息列表
    paths_list, api_list = api_list_process(retrieve_list)

    # 搭建qianfan服务请求一言模型
    f = qianfan.ChatCompletion(model="ERNIE-Functions-8K")
    msgs = qianfan.QfMessages()
    msgs.append(query, role="user")

    relevant_APIs = []
    answer = {"query": query, "query_id": query_id}
    print(f"用户query：{query}")
    # 超出10轮就退出
    n = 0
    while n < 10:
        # 请求一言模型失败也退出
        try:
            response, func_name, kwargs = function_request_yiyan_qianfan(
                f, msgs, api_list
            )  # 这里找一个最合适的API
        except Exception as e:
            print(e)
            break

        if isinstance(response, str):
            print(f"智能体回答：{response}")
            answer["result"] = response  # <--------------- 只能从这里退出
            break
        relevant_APIs.append({"api_name": func_name, "required_parameters": kwargs})
        print(f"调用函数：{func_name}, 参数：{kwargs}")

        try:
            paths = next(
                item["paths"] for item in paths_list if item["name"] == func_name
            )  # '/plugins?id=14'
        except StopIteration:
            print("模型由于幻觉生成不存在的工具")
            continue

        func_response = request_plugin(paths, kwargs)
        func_content = json.dumps({"return": func_response})
        print(f"函数返回：{func_response}")

        msgs.append(response, role="assistant")
        msgs.append(func_content, role="function")
        n += 1

        # 防止请求一言频率过高, 休眠0.5s
        time.sleep(0.5)

    answer["relevant APIs"] = relevant_APIs
    if not answer.get("result"):
        answer["result"] = "抱歉, 无法回答您的问题。"
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(answer, ensure_ascii=False) + "\n")


def tool_use_aistudio(query, query_id, api_path, save_path, topK):
    """
    通过调用API获取相关函数, 并使用聊天补全模型生成回答

    Args:
        query: 用户输入的查询语句
        api_path: API的查询路径
        save_path: 保存答案的文件路径
        topK: 返回的召回的API数量

    Returns:
        None

    """
    description_list, description_dict = get_tools_description(api_path)

    # 关键词命中优先级最高
    keywords_hit = hit_keywords(query)
    # 做召回
    retrieve_list = retrieve_api(query, topK, description_list)

    topk_api_id = set(keywords_hit) | set(retrieve_list)
    retrieve_list = [description_dict[description_list[id]] for id in topk_api_id]

    # 对API列表进行处理, 获取url路径列表和标准API信息列表
    paths_list, api_list = api_list_process(retrieve_list)

    #
    PREFIX = "你是一个AI助手, 很擅长使用不同的工具来解决用户query中的问题, 请记住, 用户的query中可能包含多个问题, 请根据思维链的提示依次完整回答,以下是用户的query:\n"
    messages = [
        # You are an AI assistant that can call functions from the OpenAI API and use them to generate responses based on user input.
        {
            "role": "user",
            "content": f"{PREFIX}\n{query}",
        }
    ]

    relevant_APIs = []
    answer = {"query": query, "query_id": query_id}
    # print(f"用户query：{query}")
    # 超出10轮就退出
    n = 0
    while n < 10:
        # 请求一言模型失败也退出
        try:
            response, func_name, kwargs = function_request_yiyan_aistudio(
                None, messages, api_list
            )  # 这里找一个最合适的API
        except Exception as e:
            print(e)
            break

        if isinstance(response, str):
            print(f"[智能体回答]：{response}")
            answer["result"] = response  # <--------------- 只能从这里退出
            break
        relevant_APIs.append({"api_name": func_name, "required_parameters": kwargs})
        print(f"[调用函数]：{func_name}, 参数：{kwargs}")

        try:
            paths = next(
                item["paths"] for item in paths_list if item["name"] == func_name
            )  # '/plugins?id=14'
        except StopIteration:
            print("模型由于幻觉生成不存在的工具")
            continue

        func_response = request_plugin(paths, kwargs)
        func_content = json.dumps({"return": func_response})
        print(f"[函数返回]：{func_response}")

        assert hasattr(response, "function_call")
        function_call = response.function_call

        assert "thoughts" in response.get_result()
        print("[Thought]:", response.get_result()["thoughts"])

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

    answer["relevant APIs"] = relevant_APIs
    if not answer.get("result"):
        answer["result"] = "抱歉, 无法回答您的问题。"
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(answer, ensure_ascii=False) + "\n")


def tool_use_aistudio_ntimes(query, query_id, api_path, save_path, topK):
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
    sub_query_list = query_split(query)
    description_list, description_dict = get_tools_description(api_path)

    relevant_APIs = []
    answer = {"query": query, "query_id": query_id}
    # print(f"用户query：{query}")

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

        messages = [
            {
                "role": "user",
                "content": f"{sub_query}",
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
                print(e)
                break

            if isinstance(response, str):
                print(f"[智能体回答]：{response}")
                # answer["result"] = response  # <--------------- 只能从这里退出
                sub_query_answer[idx] = response
                break
            relevant_APIs.append({"api_name": func_name, "required_parameters": kwargs})
            print(f"[调用函数]：{func_name}, 参数：{kwargs}")

            try:
                paths = next(
                    item["paths"] for item in paths_list if item["name"] == func_name
                )  # '/plugins?id=14'
            except StopIteration:
                print("模型由于幻觉生成不存在的工具")
                continue

            func_response = request_plugin(paths, kwargs)
            func_content = json.dumps({"return": func_response})
            print(f"[函数返回]：{func_response}")

            assert hasattr(response, "function_call")
            function_call = response.function_call

            assert "thoughts" in response.get_result()
            print("[Thought]:", response.get_result()["thoughts"])

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
        query, sub_query_list, sub_query_answer, model="ernie-3.5"
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
            query = line["query"]
            query_id = line["qid"]
            tool_use_qianfan(query, query_id, api_path, save_path, topK)
            # tool_use_aistudio(query, query_id, api_path, save_path, topK)
            # tool_use_aistudio_ntimes(query, query_id, api_path, save_path, topK)


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

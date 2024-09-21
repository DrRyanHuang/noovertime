import requests
import time
import json
import qianfan


def truncate_json(data, num_keys):
    truncated_data = {}
    count = 0
    for key in list(data.keys())[::-1]:
        if count >= num_keys:
            break
        truncated_data[key] = data[key]
        count += 1
    return truncated_data


def function_request_yiyan(f, msgs, func_list):
    """
    发送请求到一言大模型，获取返回结果。

    Args:
        f: 一言API的访问对象。
        msgs: 请求消息列表.
        func_list: 请求中需要调用的API列表。

    Returns:
        返回值为一个包含三个元素的元组:
        - response: 一言大模型返回的响应结果。
        - func_name: 响应结果中调用的函数名，为str类型。
        - kwargs: 响应结果中调用的函数的参数，为dict类型。

    """
    response = f.do(messages=msgs, functions=func_list)
    if response["body"]["result"]:
        return response["body"]["result"], "", ""
    func_call_result = response["function_call"]
    func_name = func_call_result["name"]
    kwargs = json.loads(func_call_result["arguments"])
    return response, func_name, kwargs


def request_plugin(paths, params):
    url = "http://match-meg-search-agent-api.cloud-to-idc.aistudio.internal" + paths
    try:
        response = requests.get(url, params=params).json()
        # 工具返回结果过长，做截断处理
        if len(str(response)) > 1000:
            response = truncate_json(response, 2)
    except Exception:
        response = "error：404"
    return response


def insert2dict(keyword, __dict, tool_name):
    if keyword in __dict:
        if not tool_name in __dict[keyword]:
            __dict[keyword].append(tool_name)
    else:
        __dict[keyword] = [tool_name]


def add1_2dict(keyword, __dict):
    if keyword in __dict:
        __dict[keyword] += 1
    else:
        __dict[keyword] = 1

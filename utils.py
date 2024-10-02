import requests
import time
import json
import qianfan
import erniebot


def truncate_json(data, num_keys):
    truncated_data = {}
    count = 0
    for key in list(data.keys())[::-1]:
        if count >= num_keys:
            break
        truncated_data[key] = data[key]
        count += 1
    return truncated_data


def function_request_yiyan_qianfan(f, msgs, func_list):
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


def function_request_yiyan_aistudio(f, msgs, func_list, model="ernie-3.5"):
    """
    发送请求到一言大模型，获取返回结果。

    Args:
        f: 一言API的访问对象。
        msgs: 请求消息列表.
        func_list: 请求中需要调用的API列表。

        ernie-lite | ernie-speed 不支持 functions

    Returns:
        返回值为一个包含三个元素的元组:
        - response: 一言大模型返回的响应结果。
        - func_name: 响应结果中调用的函数名，为str类型。
        - kwargs: 响应结果中调用的函数的参数，为dict类型。

    """

    response = erniebot.ChatCompletion.create(
        model=model, messages=msgs, functions=func_list, stream=False
    )

    # print(response.get_result())
    if response["result"]:
        return response["result"], "", ""

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


def run_once(func):
    """
    使函数只运行一次，并将结果缓存下来，之后再次调用时直接返回缓存结果。
    https://github.com/Obmutescence/4DPocket/blob/main/python/utils/decorator.py

    Args:
        func (Callable): 需要被装饰的函数，无返回值或返回任意类型均可。

    Returns:
        Callable: 返回一个新的函数，该函数只会在第一次调用时执行原始函数，并将结果缓存下来。
            之后再次调用时直接返回缓存结果。

    """

    def wrapper(*args, **kwargs):
        if not wrapper._has_run:
            wrapper._has_run = True
            wrapper._result = func(*args, **kwargs)
        return wrapper._result

    wrapper._has_run = False
    return wrapper


def api_list_process(retrieve_list):
    """
    从给定的API列表中提取url路径列表和标准API信息列表。

    Args:
        retrieve_list (List[Dict]): API列表, 每个元素是一个字典, 包含标准API信息和url路径信息。

    Returns:
        Tuple[List[Dict], List[Dict]]: 包含APIurl路径和API信息的两个列表的元组。
            - paths_list (List[Dict]): 包含APIurl路径信息的列表, 每个元素是一个字典, 包含API名称和路径信息。
            - api_list (List[Dict]): 包含API信息的列表, 每个元素是一个字典, 包含API信息中除路径外的所有字段。

    """
    paths_list = [{"name": api["name"], "paths": api["paths"]} for api in retrieve_list]
    api_list = [{k: v for k, v in api.items() if k != "paths"} for api in retrieve_list]
    return paths_list, api_list

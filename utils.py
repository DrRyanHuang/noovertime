import requests
import time
import json
import sys
import qianfan
import erniebot
from lunar_python import Solar
import re
import datetime
import unittest
import timeit


class TimerContext:
    def __init__(self, name):
        self.name = name
        self.total_time = 0

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = timeit.default_timer()
        execution_time = end_time - self.start_time
        self.total_time += execution_time
        print(f"{self.name} 执行时间: {execution_time:.5f} 秒")

    def print_total_time(self):
        print(f"{self.name} 总执行时间: {self.total_time:.5f} 秒")


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
        model=model,
        messages=msgs,
        functions=func_list,
        stream=False,
        # TODO: 温度没改
        # https://ernie-bot-agent.readthedocs.io/zh-cn/latest/sdk/api_reference/chat_completion/#_1
        # temperature=0.05,
        request_timeout=600,
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


class DotDict:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __getattr__(self, attr):
        return self.__dict__.get(attr)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __repr__(self):
        return repr(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


class DotDictTest1(unittest.TestCase):

    def setUp(self):
        self.dot_dict = DotDict({"a": 1, "b": 2}, c=3)

    def test_init(self):
        self.assertEqual(self.dot_dict.a, 1)
        self.assertEqual(self.dot_dict.b, 2)
        self.assertEqual(self.dot_dict.c, 3)

    def test_setattr(self):
        self.dot_dict.d = 4
        self.assertEqual(self.dot_dict.d, 4)

    def test_getitem(self):
        self.assertEqual(self.dot_dict["a"], 1)
        with self.assertRaises(KeyError):
            _ = self.dot_dict["e"]

    def test_get(self):
        self.assertEqual(self.dot_dict.get("a"), 1)
        self.assertEqual(self.dot_dict.get("e", 5), 5)

    def test_repr(self):
        self.assertEqual(repr(self.dot_dict), "{'a': 1, 'b': 2, 'c': 3}")

    def test_iter(self):
        keys = [key for key in self.dot_dict]
        self.assertListEqual(keys, ["a", "b", "c"])

    def test_len(self):
        self.assertEqual(len(self.dot_dict), 3)


class DotDictTest2(unittest.TestCase):

    def setUp(self):
        # 创建一个 DotDict 实例，用于每个测试方法
        self.dot_dict = DotDict({"key1": "value1"}, key2="value2")

    def test_init(self):
        # 测试 __init__ 方法是否正确初始化了字典
        self.assertEqual(self.dot_dict.key1, "value1")
        self.assertEqual(self.dot_dict.key2, "value2")

    def test_getattr(self):
        # 测试 __getattr__ 方法是否正确返回字典中的值
        self.assertEqual(self.dot_dict.key1, "value1")
        # 测试不存在的键返回 None，而不是抛出异常
        self.assertIsNone(self.dot_dict.nonexistent_key)

    def test_setattr(self):
        # 测试 __setattr__ 方法是否正确设置字典中的值
        self.dot_dict.key3 = "value3"
        self.assertEqual(self.dot_dict.key3, "value3")

    def test_getitem(self):
        # 测试 __getitem__ 方法是否正确返回字典中的值
        self.assertEqual(self.dot_dict["key1"], "value1")
        # 测试不存在的键会抛出 KeyError
        with self.assertRaises(KeyError):
            _ = self.dot_dict["nonexistent_key"]

    def test_setitem(self):
        # 测试 __setitem__ 方法是否正确设置字典中的值
        self.dot_dict["key3"] = "value3"
        self.assertEqual(self.dot_dict["key3"], "value3")

    def test_get(self):
        # 测试 get 方法是否正确返回字典中的值，以及默认值
        self.assertEqual(self.dot_dict.get("key1"), "value1")
        self.assertEqual(
            self.dot_dict.get("nonexistent_key", "default_value"), "default_value"
        )

    def test_repr(self):
        # 测试 __repr__ 方法是否正确返回字典的字符串表示
        expected_repr = "{'key1': 'value1', 'key2': 'value2'}"
        self.assertEqual(repr(self.dot_dict), expected_repr)

    def test_iter(self):
        # 测试 __iter__ 方法是否正确迭代字典的键
        keys = [key for key in self.dot_dict]
        self.assertListEqual(keys, ["key1", "key2"])

    def test_len(self):
        # 测试 __len__ 方法是否正确返回字典的长度
        self.assertEqual(len(self.dot_dict), 2)


if __name__ == "__main__":
    unittest.main()


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


# ---------- 由于农历相关的API失灵，这里进行Python库的手动替换 ----------
def convert_to_lunar_date(year, month, day):
    solar = Solar.fromYmd(year, month, day)
    lunar = solar.getLunar()
    return lunar.getYear(), lunar.getMonth(), lunar.getDay()


def convert_to_lunar_date2str(*date_tuple):
    solar = Solar.fromYmd(date_tuple[0], date_tuple[1], date_tuple[2])
    lunar = solar.getLunar()
    year = lunar.getYear()
    month = lunar.getMonthInChinese()
    day = lunar.getDayInChinese()
    return "农历{}月{}".format(month, day)


if __name__ == "__main__":

    print(convert_to_lunar_date(2024, 10, 7))  # 输出: (2024, 9, 5)
    print(convert_to_lunar_date2str(2024, 10, 7))  # 输出: 农历九月初五


def find_and_convert_date(s):
    # 将含有日期的字符串转化为标准格式
    patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}号",
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{4}/\d{1,2}/\d{1,2}",
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            date_str = match.group()
            for fmt in ["%Y年%m月%d号", "%Y年%m月%d日", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    return datetime.datetime.strptime(date_str, fmt).strftime(
                        "%Y-%m-%d"
                    )
                except ValueError:
                    pass
    # raise ValueError('no valid date format found')
    return ""


if __name__ == "__main__":
    print(find_and_convert_date("2024年9月8号，驾车去北京"))  # 输出: 2024-09-08


def get_weekday(year, month, day):
    # 传入日期，拿到其是星期几
    day_number = datetime.date(year, month, day).weekday()
    days = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return days[day_number]


if __name__ == "__main__":
    # 测试这个函数
    print(get_weekday(2024, 9, 1))  # 输出: 星期日

color_head = {
    # 基本颜色
    "red": "[31m",
    "green": "[32m",
    "yellow": "[33m",
    "blue": "[34m",
    "magenta": "[35m",
    "cyan": "[36m",
    "white": "[37m",
    "black": "[30m",
    # 高亮颜色
    "bright_red": "[91m",
    "bright_green": "[92m",
    "bright_yellow": "[93m",
    "bright_blue": "[94m",
    "bright_magenta": "[95m",
    "bright_cyan": "[96m",
    "bright_white": "[97m",
    # 文本格式
    "reset": "[0m",  # 重置为默认设置
    "bold": "[1m",  # 设置为粗体
    "underline": "[4m",  # 设置下划线
    "blink": "[5m",  # 设置为闪烁
    "reverse": "[7m",  # 反显
    "conceal": "[8m",  # 隐藏
}


# 一个简单的打 log 的函数，同时输出到控制台和文件
def pp_print(*args, file_path="logs/log.txt", color="red", **kwargs):
    print(f"\033{color_head[color]}")
    print(*args, file=sys.stdout, **kwargs)
    print("\033[0m")
    with open(file_path, "a") as f:
        print(*args, file=f, **kwargs)


def find_metas_in_s(s, metas):
    found_items = [term for term in metas if term in s]
    return found_items


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

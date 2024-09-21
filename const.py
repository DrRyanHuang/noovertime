import json
from pprint import pprint
from utils import insert2dict

SIGNIFICANT_MAPPINGS = {
    "限行尾号": ["bd_gov_xianxing"],
    "汉语词典": ["get_hy_zici", "chinese_character_word_info_query"],
    "农历": ["bd_calendar"],
    "阴历": ["bd_calendar"],
    "拥堵指数": ["get_congestionindex_data_from_poi"],
    "OPPO": ["oppo_info_search_service"],
    "VIVO": ["vivo_info_search_service"],
    "汽油": ["baidu_fule_price"],
    "柴油": ["baidu_fule_price"],
    "温度": ["baidu_muti_weather"],
    "对联": ["get_hy_sentence_couplet"],
}

CHINESE_TRADITIONAL_FESTIVALS = """
除夕
春节
元宵
清明
端午
七夕
中秋
重阳
腊八
"""

CHINESE_OTHER_FESTIVALS = """
愚人节
劳动节
儿童节
建军节
建党节
国庆
"""

SOLAR_TERMS24 = """
立春,雨水,惊蛰,春分,清明,谷雨
立夏,小满,芒种,夏至,小暑,大暑
立秋,处暑,白露,秋分,寒露,霜降
立冬,小雪,大雪,冬至,小寒,大寒
""".strip().replace(
    "\n", ","
)


EXPRESS_DELIVERY = """
快递,顺丰,圆通,申通,韵达,百世,中通,德邦,邮政,EMS
"""


# 绑定 {关键字, 工具名字} 到 SIGNIFICANT_MAPPINGS 中
def bind_keyword2SIGNIFICANT_MAPPINGS(*args, tool_name=None, sep="\n"):
    if tool_name is None:
        raise ValueError("请传入关键字")
    else:
        for arg in args:
            if isinstance(arg, str):
                if sep in arg:
                    for keyword in arg.strip().split(sep):
                        # SIGNIFICANT_MAPPINGS[keyword] = tool_name
                        insert2dict(keyword, SIGNIFICANT_MAPPINGS, tool_name)
                else:
                    # SIGNIFICANT_MAPPINGS[arg] = tool_name
                    insert2dict(keyword, SIGNIFICANT_MAPPINGS, tool_name)


bind_keyword2SIGNIFICANT_MAPPINGS(
    CHINESE_TRADITIONAL_FESTIVALS, CHINESE_OTHER_FESTIVALS, tool_name="bd_calendar"
)
bind_keyword2SIGNIFICANT_MAPPINGS(SOLAR_TERMS24, tool_name="bd_calendar", sep=",")

bind_keyword2SIGNIFICANT_MAPPINGS(
    EXPRESS_DELIVERY, tool_name="get_express_info", sep=","
)

ID2TOOL = {}
TOOL2ID = {}
with open("api_list.json") as f:
    for __idx, line in enumerate(f.readlines()):
        tool_dict = json.loads(line)
        ID2TOOL[__idx] = tool_dict["name"]
        TOOL2ID[tool_dict["name"]] = __idx

# pprint(SIGNIFICANT_MAPPINGS)
# pprint(ID2TOOL)
# pprint(TOOL2ID)

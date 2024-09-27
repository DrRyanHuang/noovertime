UESR_INTENTION_ANALYSIS_PROMPT = dict(content="""""")


FINETINE_DATA_GENERATOR_PROMPT = dict(
    content="""
在工具匹配过程中, 用户query和工具描述存在匹配关系, 请你根据以下示例, 和给定的输入生成出{data_num}条数据

【示例】
【工具描述】
{demo_tool_desc}

【用户query】
{demo_user_query}

【给定工具描述】
{input_tool_desc}

请按照字符串列表的形式返回所生成的用户query, 如"[str, str, str]"
""".strip()
)

INTENT_DISASSEMBLE_PROMPT = dict(
    content="""
你是一名任务分析工程师，现在需要对用户提出的问题进行解析，要求将用户的一系列问题拆分为独立的问题：

**要求**
1、问题拆分需要具备逻辑性。
2、问题需要补全各个要素。

**样例输入-1**
<用户问题>：
我计划2024年9月去青岛旅游，想了解八大关景点的门票价格和开放时间，再帮我找一家最便宜的饺子馆的名字。

**样例输出-1**
<拆分问题列表>：
[2024年9月青岛的八大关景点的门票价格和开放时间。]
[2024年9月青岛的最便宜的饺子馆的名字。]

**样例输入-2**
<用户问题>：
我想知道腾讯股票的代码是多少，百度呢？

**样例输出-2**
<拆分问题列表>：
[腾讯股票的代码]
[百度股票的代码]

**现在输入**
<用户问题>：
{input_query}

**要求**
1、生成的<拆分问题列表>需要在给定的<需求>范围内，不可以延伸出其他的问题
2、<拆分问题列表>中的问题需要齐全。

**限制**
1、不能额外生成未见过的问题。

请仿照样例，严格遵照要求和限制，分析<用户问题>，（用户问题是：{input_query}），生成简洁通顺，符合逻辑性的<拆分问题列表>
""".strip()
)
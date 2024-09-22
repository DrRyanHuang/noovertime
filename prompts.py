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

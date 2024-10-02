import re
import json
from prompts import COMBINE_SUBQUERY_ANSWER_PROMPT
from config import AISTUDIO_AK

import erniebot
from tqdm import tqdm


def answer_combine(origin_query, sub_querys, answers, model="ernie-3.5"):

    content = COMBINE_SUBQUERY_ANSWER_PROMPT["content"].replace("{query}", origin_query)
    content = content.replace("{req_list}", "\n".join(answers))

    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )

    # print(response.get_result())
    response = response.get_result()

    return response


if __name__ == "__main__":

    erniebot.api_type = "aistudio"
    erniebot.access_token = AISTUDIO_AK

    origin_query = "2024年9月10号从上海开车去苏州，最短多长时间，沿途过路费最少多少钱？当天苏州的最高温度如何？"

    sub_query_list = [
        "2024年9月10号从上海开车到苏州的最短用时是多少？",
        "2024年9月10号从上海开车到苏州的沿途过路费最少需要多少钱？",
        "2024年9月10号苏州的最高温度是多少？",
    ]

    answers = [
        "根据您的要求，我查询了2024年9月10号从上海到苏州的开车最短用时。不过，由于'ticket_info_query'工具并未直接提供具体日期的最短用时信息，而是给出了一个示例日期（10月1日）的结果作为参考。该示例显示，从上海到苏州的开车最短用时为1时30分。但请注意，这只是一个大致的参考时间，实际用时可能会因交通状况、天气等因素而有所变化。如果您需要更精确的信息，建议查询实时的导航软件或咨询当地的交通部门。",
        "根据您的要求，我查询了2024年9月10号从上海到苏州的汽车票信息。但是，工具返回的信息是10月1日的数据，显示的是直达的票价和耗时，其中最低票价为45元，最短耗时为1时30分。由于我无法直接获取特定日期的过路费信息，这个票价可能包含了部分过路费，但具体过路费金额可能因实际路况和收费站政策而有所不同。如果您需要更准确的过路费信息，建议您查询相关的高速公路收费站或使用导航软件进行实时计算。",
        "根据您提供的信息，2024年9月10号苏州的最高温度为35度。但是，我注意到工具返回的结果中，开始日期和结束日期都是2024年9月30日，这可能是由于工具的限制或数据的问题。不过，由于您只关心9月10号的温度，我们可以假设这一天的温度与返回结果中的温度相近。如果您需要更精确的信息，建议您查看当地的气象报告或联系当地气象部门。",
    ]

    your_answer = answer_combine(origin_query, sub_query_list, answers)
    print(your_answer)

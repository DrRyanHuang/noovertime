<!-- English | [简体中文](./README_cn.md) -->

<div align="center">
<!-- 标题 -->

<h1 align="center">
  - NO-OVERTIME - 
</h1>
<!-- star数, fork数, pulls数, issues数, contributors数, 开源协议 -->

<a href="https://github.com/DrRyanHuang/noovertime/stargazers"><img src="https://img.shields.io/github/stars/DrRyanHuang/noovertime" alt="Stars Badge"/></a>
<a href="https://github.com/DrRyanHuang/noovertime/network/members"><img src="https://img.shields.io/github/forks/DrRyanHuang/noovertime" alt="Forks Badge"/></a>
<br/>
<a href="https://github.com/DrRyanHuang/noovertime/pulls"><img src="https://img.shields.io/github/issues-pr/DrRyanHuang/noovertime" alt="Pull Requests Badge"/></a>
<a href="https://github.com/DrRyanHuang/noovertime/issues"><img src="https://img.shields.io/github/issues/DrRyanHuang/noovertime" alt="Issues Badge"/></a>
<a href="https://github.com/DrRyanHuang/noovertime/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/DrRyanHuang/noovertime?color=2b9348"></a>
<a href="https://github.com/DrRyanHuang/noovertime/blob/master/LICENSE"><img src="https://img.shields.io/github/license/DrRyanHuang/noovertime?color=2b9348" alt="License Badge"/></a>

<!-- logo -->

<!-- 
<img alt="LOGO" src="logo/Frieren_sleep.png" width="30%"> </img>
<br/>
<i>Loved the project? Please consider forking the project to help it improve!</i>🌟
-->

</div>

本代码库是 **百度和英伟达** 共同发起的“2024 百度搜索・文心智能体创新大赛”技术赛道赛题的代码库。
该比赛旨在挑战如何利用 **文心一言 (ERNIE) 系列大模型** 构建高智能的 **AI Agent**，参赛者需要开发一个能够从海量工具集中（49个API）精准检索并调度工具，以解决旅游、金融、天气等跨领域复杂问题的智能系统。

* **比赛网址：**[https://aistudio.baidu.com/competition/detail/1235/0/task-definition](https://aistudio.baidu.com/competition/detail/1235/0/introduction)
* **团队成绩**：14/1008。

以下为比赛简要介绍。

### 1. 任务介绍

 **主题**：基于多工具调用的开放域问答智能体构建。主要包含两个核心阶段：

* **工具召回 (Retrieval)：** 面对用户的复杂提问，利用 **ERNIE 基座模型** 从 49 个备选 API 中筛选出最相关的工具子集（支持 NVIDIA 插件加速）。
* **工具编排 (Orchestration)：** 调度选中的 API，通过单步或多步调用获取信息，最终生成准确的开放域问题答案。

### 2. 技术约束

* **大模型基座：** 必须使用百度 **eb-系列 (ERNIE)** 模型。
* **召回模块：** 必须使用 **ERNIE 基座** 开发，允许使用 NVIDIA 加速套件进行推理优化。
* **数据利用：** 官方提供 100 道赛题样例及 49 个标准 API 定义（含名称、功能描述及参数列表），鼓励选手自行构建训练集。

### 3. API 样例参考

工具以标准 JSON 格式给出，包含功能描述及参数约束：

```json
{
    "name": "ticket_info_query",
    "description": "获取火车、飞机、汽车的最短耗时和最低票价",
    "parameters": {
        "departure": "出发地",
        "destination": "目的地",
        "travel_mode": "出行方式"
    }
}
```
### 4. 评估维度

总分由 **答案准确性**、**工具调用完整性** 和 **效率** 加权计算得出：

| 指标 | 权重 | 定义 |
| --- | --- | --- |
| **Correctness** | 50% | 输出答案与标准答案之间的 Token 粒度 F1 值。 |
| **Comprehensiveness** | 30% | 正确工具的召回率（正确调用的 API 占标准 API 的比例）。 |
| **Efficiency** | 20% | 工具调用的准确率（正确调用的 API 占总调用次数的比例）。 |



### 5. 优化方向建议

为了超越基础 Baseline，可以从以下三个维度进行突破：

* **检索策略微调：** 构造「Query-API」特征对，对 ERNIE 模型进行微调，提升 Top-K 检索的覆盖率。
* **推理规划增强：** 放弃简单的线性调用，引入 **ReAct**、**DFS (深度优先搜索)** 或 **Chain-of-Thought** 等复杂规划算法。
* **多智能体协作：** 设计更精巧的 Prompt，或利用不同的文心大模型构建 **Multi-Agent 系统** 分解任务。


1. inference.py 是推理的入口，inference.sh 则对此做了参数的封装。python 的执行需要严格仿照 inference.sh 中所给出的范式。
2. api_list.json 是全部的API工具，已转成标准的API格式，其中 "paths" 字段用于访问 API。
3. model 文件是本次大赛的参考 retrieval model，用于检索用户 query 对应的 APIs。
4. dataset.json 是本次的赛题数据集，选手们可以基于赛题和api_list，构建自己的数据集训练一个 retrieval model。
5. spyder.py 是构建知识库的爬虫

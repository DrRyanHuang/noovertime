<!-- English | [简体中文](./README_cn.md) -->

<div align="center">
<!-- 标题 -->

<h1 align="center">
  - NO-OVERTIME - 
</h1>
noovertime
<!-- star数, fork数, pulls数, issues数, contributors数, 开源协议 -->

<a href="https://github.com/DrRyanHuang/bangumi-anime/stargazers"><img src="https://img.shields.io/github/stars/DrRyanHuang/bangumi-anime" alt="Stars Badge"/></a>
<a href="https://github.com/DrRyanHuang/bangumi-anime/network/members"><img src="https://img.shields.io/github/forks/DrRyanHuang/bangumi-anime" alt="Forks Badge"/></a>
<br/>
<a href="https://github.com/DrRyanHuang/bangumi-anime/pulls"><img src="https://img.shields.io/github/issues-pr/DrRyanHuang/bangumi-anime" alt="Pull Requests Badge"/></a>
<a href="https://github.com/DrRyanHuang/bangumi-anime/issues"><img src="https://img.shields.io/github/issues/DrRyanHuang/bangumi-anime" alt="Issues Badge"/></a>
<a href="https://github.com/DrRyanHuang/bangumi-anime/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/DrRyanHuang/bangumi-anime?color=2b9348"></a>
<a href="https://github.com/DrRyanHuang/bangumi-anime/blob/master/LICENSE"><img src="https://img.shields.io/github/license/DrRyanHuang/bangumi-anime?color=2b9348" alt="License Badge"/></a>

<!-- logo -->

<!-- 
<img alt="LOGO" src="logo/Frieren_sleep.png" width="30%"> </img>
<br/>
<i>Loved the project? Please consider forking the project to help it improve!</i>🌟
-->

</div>

1. inference.py 是推理的入口，inference.sh 则对此做了参数的封装。python 的执行需要严格仿照 inference.sh 中所给出的范式。
2. api_list.json 是全部的API工具，已转成标准的API格式，其中 "paths" 字段用于访问 API。
3. model 文件是本次大赛的参考 retrieval model，用于检索用户 query 对应的 APIs。
4. dataset.json 是本次的赛题数据集，选手们可以基于赛题和api_list，构建自己的数据集训练一个 retrieval model。
5. spyder.py 是构建知识库的爬虫
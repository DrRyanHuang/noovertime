import erniebot
from milvus import default_server
from pymilvus import connections
from pymilvus import utility, Collection
from pymilvus import CollectionSchema, FieldSchema, DataType
from pprint import pprint
from config import AISTUDIO_AK
import time, os, pickle
import numpy as np
from utils import run_once


erniebot.api_type = "aistudio"
erniebot.access_token = AISTUDIO_AK


# 请注意只能启动一次，重复启动会报错，如需重启数据库，请先重启环境。
try:
    default_server.start()
except:
    default_server.cleanup()
    default_server.start()
connections.connect(host="127.0.0.1", port=default_server.listen_port)


# -------------------- 创建数据表 --------------------
# 创建一个数据集合（可以理解为Mysql里面的数据表），里面包含三个字段，
# 分别是`answer_id`（自增数据ID）、`answer`（回答文本）以及最重要的`answer_vector`（回答文本的对应向量），
# 使用文心百中语义模型，其输出的数据维度为384，
# 如果是其他模型，请遵循其他模型对应的输出维度。然后把这个集合记为"qadb"。


answer_id = FieldSchema(
    name="answer_id", dtype=DataType.INT64, is_primary=True, auto_id=True
)
answer = FieldSchema(
    name="answer",
    dtype=DataType.VARCHAR,
    max_length=1024,
)
answer_vector = FieldSchema(name="answer_vector", dtype=DataType.FLOAT_VECTOR, dim=384)

schema = CollectionSchema(
    fields=[answer_id, answer, answer_vector], description="vector data"
)

collection_name = "qadb"

collection = Collection(
    name=collection_name, schema=schema, using="default", shards_num=2
)


# ## 创建索引和加载数据库
# 创建集合后，还需要对该集合的answer_vector向量列添加索引，以进行后续的向量运算。这里使用欧氏距离和`IVF_FLATIVF_FLAT`算法对向量建立索引。


index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},
}

collection.create_index(field_name="answer_vector", index_params=index_params)
utility.index_building_progress(collection_name)

collection.load()


# 单例模式
class milvusRecoder:
    _instance = None  # 私有类变量，用于存储类的唯一实例

    # 用来记录哪些文件其实已经被录入了
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.recorder_file = "recorder.txt"
            cls._instance.recorder = set()
            cls._instance.get_recorders()  # 初始化时加载已有记录
        return cls._instance

    def _load_recorders(self):
        if not os.path.exists(self.recorder_file):
            return set()
        # 使用 with 语句确保文件正确关闭
        with open(self.recorder_file, "r") as file:
            return set(file.read().strip().split("\n"))

    def get_recorders(self):
        # 如果需要动态更新 recorder 集合，可以调用此方法
        # 否则，可以直接访问 self.recorder 属性
        self.recorder = self._load_recorders()

    def add_recorder(self, text):
        if text in self.recorder:
            return
        self.recorder.add(text)
        with open(self.recorder_file, "a+") as f:
            f.write(text + "\n")


RECORDER = milvusRecoder()


class PoemSplitter:
    _instance = None  # 私有类变量，用于存储类的唯一实例
    CHUNK_SIZE = 300  # 每段的最大长度

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, poem_file):
        with open(poem_file) as f:
            content = f.read()
        return self.__split(content)

    @classmethod
    def __sentence_split(cls, sentence, chunk_size=300, sep=250):

        # 切分的时候要重叠

        start = 0
        end = start + chunk_size
        split_result = []
        while start < len(sentence):
            split_result.append(sentence[start:end])
            start += sep
            end = start + chunk_size

        return split_result

    def __split(self, content):

        content = content.replace("\u3000", " ")
        __result_list = content.strip().split("\n" * 5)
        __result_list = [item.strip() for item in __result_list]

        result_list = []
        for item in __result_list:
            result_list += self.__sentence_split(item)

        result_list = list(filter(lambda x: len(x) > 10, result_list))

        return result_list


def get_embedding(text):
    # 传入文本, 拿其 embedding
    response = erniebot.Embedding.create(model="ernie-text-embedding", input=[text])

    return response.get_result()


def combine_answer_question(contents, question, temperature=0.9, top_p=0.9):

    erniebotInput = (
        "使用以下文段来回答最后的问题。仅根据给定的文段生成答案。如果你在给定的文段中没有找到任何与问题相关的信息，就说你不知道，不要试图编造答案。保持你的答案富有表现力。用户最后的问题是："
        + question
        + "。给定的文段是："
        + contents
    )

    response = erniebot.ChatCompletion.create(
        model="ernie-4.0",
        messages=[{"role": "user", "content": erniebotInput}],
        temperature=temperature,
        top_p=top_p,
    )
    return response.get_result()


@run_once
def get_global_dict(path="./global_dict.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            global_dict = pickle.load(f)
    else:
        global_dict = {}
        with open(path, "wb") as f:
            pickle.dump(global_dict, f)

    return global_dict


def save_global_dict(global_dict, path="./global_dict.pkl"):
    with open(path, "wb") as f:
        pickle.dump(global_dict, f)


def insert2milvus(content_list, collection):

    global_dict = get_global_dict()

    # 数据库、数据集合和索引都建立之后，就可以向里面插入数据
    for content in content_list:

        if content in global_dict:
            continue

        contentEmbedding = get_embedding(content)
        data = [[content], contentEmbedding]

        global_dict[content] = np.array(contentEmbedding)

        collection.insert(data)
        time.sleep(0.3)  # 加延时，防止embedding接口超QPS


SEARCH_PARAMS = {
    "metric_type": "L2",
    "offset": 0,
    "ignore_growing": False,
    "params": {"nprobe": 10},
}


def get_search_results(question, collection, limit=3):

    # 得到检索的结果

    qEmbedding = get_embedding(question)

    results = collection.search(
        data=qEmbedding,
        anns_field="answer_vector",
        param=SEARCH_PARAMS,
        limit=limit,
        expr=None,
        output_fields=["answer"],
        consistency_level="Strong",
    )

    answer_list = []
    for idx in range(limit):
        answer = results[0][idx].entity.get("answer")
        # print("answer: ", answer)
        answer_list.append(answer)

    return "\n".join(answer_list)


if __name__ == "__main__":
    recorder = milvusRecoder()
    recorder.add_recorder("hjk")

    splitter = PoemSplitter()
    contents = splitter("poem/辛弃疾/破阵子·为陈同甫赋壮词以寄之.txt")

    pprint([len(sentence) for sentence in contents])
    pprint([sentence[:20] for sentence in contents])

    insert2milvus(contents, collection)

    question = "“五十弦翻塞外声”是哪位诗人写的？"
    search_results = get_search_results(question, collection)

    final_result = combine_answer_question(search_results, question)

    print(final_result)

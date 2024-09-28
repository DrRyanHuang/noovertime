import json
import functools

from paddle.io import DataLoader, BatchSampler
from paddlenlp.datasets import load_dataset
from paddlenlp.data import DataCollatorWithPadding

# 加载中文语义匹配数据集 LCQMC
# LCQMC（Large-scale Chinese Question Matching Corpus）中文语义匹配数据集,
# 基于百度知道相似问题推荐构造的通问句中文语义匹配数据集，目的是为了解决在中文领域大
# 规模问题匹配数据集的缺失。该数据集从百度知道不同领域的用户问题中抽取构建数据，数据集示例：
# ```
# query    title    label
# 喜欢打篮球的男生喜欢什么样的女生	爱打篮球的男生喜欢什么样的女生	1
# 我手机丢了，我想换个手机	我想买个新手机，求推荐	1
# 大家觉得她好看吗	大家觉得跑男好看吗	0
# 晚上睡觉带着耳机听音乐有什么害处吗？	孕妇可以戴耳机听音乐吗?	0
# ```
# 其中1表示语义相似，0表示语义不相似


DATASET_LABEL_LIST = ["0", "1"]
DATA_PATH = "data.json"


def __read(data_path, train=True, rate=0.9):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        split = int(len(lines) * rate)
        if train:
            lines = lines[:split]
        else:
            lines = lines[split:]

        for idx, line in enumerate(lines):
            line = json.loads(line.strip())
            yield line


def read_train(data_path):
    return __read(data_path, train=True)


def read_dev(data_path):
    return __read(data_path, train=False)


def __get_dataset(data_path, lazy=False):

    # data_path为read()方法的参数
    __train_ds = load_dataset(read_train, data_path=data_path, lazy=lazy)
    __dev_ds = load_dataset(read_dev, data_path=data_path, lazy=lazy)

    return __train_ds, __dev_ds


def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):

    result = tokenizer(
        text=examples["query"], text_pair=examples["title"], max_seq_len=max_seq_length
    )
    if not is_test:
        result["labels"] = examples["label"]

    # result["query"] = examples["query"]
    # result["title"] = examples["title"]
    return result


def get_dataloader(data_path, tokenizer, train_bs=64, dev_bs=128):

    __train_ds, __dev_ds = __get_dataset(data_path)

    # 通过`Dataset`的`map`函数，使用分词器将数据集中query文本和title文本拼接，从原始文本处理成模型的输入。
    # 实际训练中，根据显存大小调整批大小`batch_size`和文本最大长度`max_seq_length`。
    # 数据预处理函数，利用分词器将文本转化为整数序列
    trans_func = functools.partial(
        preprocess_function, tokenizer=tokenizer, max_seq_length=128
    )
    train_ds = __train_ds.map(trans_func)
    dev_ds = __dev_ds.map(trans_func)

    # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
    collate_fn = DataCollatorWithPadding(tokenizer)

    # BatchSampler，选择批大小和是否随机乱序，进行DataLoader
    train_batch_sampler = BatchSampler(train_ds, batch_size=train_bs, shuffle=True)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=dev_bs, shuffle=False)

    train_data_loader = DataLoader(
        dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn
    )
    dev_data_loader = DataLoader(
        dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn
    )

    return train_data_loader, dev_data_loader


if __name__ == "__main__":

    train_ds, dev_ds = __get_dataset(DATA_PATH)

    # 数据集返回为MapDataset类型
    print("数据类型:", type(train_ds))
    # label代表标签，测试集中不包含标签信息
    print("训练集样例:", train_ds[0])
    print("验证集样例:", dev_ds[0])
    # print("测试集样例:", test_ds[0])

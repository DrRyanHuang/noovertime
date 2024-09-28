import paddle
import numpy as np
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer


# 加载中文ERNIE 3.0预训练模型和分词器

# PaddleNLP中Auto模块（包括AutoModel, AutoTokenizer及各种下游任务类）提供了方便易用的接口，无需指定模型类别，
# 即可调用不同网络结构的预训练模型。PaddleNLP的预训练模型可以很容易地通过 from_pretrained 方法加载，
#
# Transformer预训练模型汇总
# https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer

# AutoModelForSequenceClassification 可用于Point-wise方式的二分类语义匹配任务，通过预训练模型获取输入文本对（query-title）的表示，
# 之后将文本表示进行分类。PaddleNLP已经实现了ERNIE 3.0预训练模型，可以通过一行代码实现ERNIE 3.0预训练模型和分词器的加载。


MODEL_NAME = "ernie-3.0-xbase-zh"


def get_model_tokenizer(model_name=MODEL_NAME, num_classes=2):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_classes=num_classes
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = (
            batch["input_ids"],
            batch["token_type_ids"],
            batch["labels"],
        )
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase, np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


@paddle.no_grad()
def inference(model, data_loader):

    model.eval()
    logits_list = []
    for batch in data_loader:
        input_ids, token_type_ids = (
            batch["input_ids"],
            batch["token_type_ids"],
        )
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits_list.append(logits)

    logits = paddle.concat(logits_list, axis=0)
    logits = F.softmax(logits, axis=-1)

    order = logits[:, -1].argsort(descending=True).cpu().numpy()
    result = logits.argmax(axis=-1).cpu().numpy()

    return order, result


def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):

    result = tokenizer(
        text=examples["query"], text_pair=examples["title"], max_seq_len=max_seq_length
    )
    if not is_test:
        result["labels"] = examples["label"]

    # result["query"] = examples["query"]
    # result["title"] = examples["title"]
    return result

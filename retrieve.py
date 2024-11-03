import paddle
from paddlenlp.transformers import AutoModel, AutoTokenizer
from model import infer, get_model_tokenizer, infer_pd_inference
from dataset import get_inference_dataloader, get_infer_dataset
from utils import TimerContext, run_once
from rank_bm25 import BM25Okapi
import jieba
import glob

USING_BASELINE = False

if USING_BASELINE:
    model_path = "model_baseline"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
else:
    model_path = "ernie_ckpt"
    model, tokenizer = get_model_tokenizer(model_path)


def get_topk_baseline(q, a, k):
    # baseline 提供的方法，将 query 和 description 转化为词向量，然后计算余弦相似度
    model.eval()

    with paddle.no_grad():
        q_inputs = tokenizer(text=q, max_seq_len=512, padding="max_length")
        query_input_ids = paddle.to_tensor([q_inputs["input_ids"]])
        query_token_type_ids = paddle.to_tensor([q_inputs["token_type_ids"]])
        _, cls_embedding_query = model(query_input_ids, query_token_type_ids)
        cls_embedding_query = cls_embedding_query.squeeze()

        a_inputs = tokenizer(text=a, max_seq_len=512, padding="max_length")
        answer_input_ids = paddle.to_tensor(a_inputs["input_ids"])
        answer_token_type_ids = paddle.to_tensor(a_inputs["token_type_ids"])
        _, cls_embedding_tool = model(answer_input_ids, answer_token_type_ids)

        cos_sim = (cls_embedding_query * cls_embedding_tool).sum(-1) / (
            paddle.linalg.norm(cls_embedding_query)
            * paddle.linalg.norm(cls_embedding_tool, axis=-1)
        )

    values, indices = paddle.topk(cos_sim, k=k)

    return indices.numpy().tolist()


def get_topk(q, a, k):
    dataset = get_infer_dataset(q, a)
    dataloader = get_inference_dataloader(dataset, tokenizer, batchsize=64)

    with TimerContext("动态图") as tc1:
        order_static, _ = infer(model, dataloader)
    # with TimerContext("Paddle Inference") as tc2:
    #     order_pd_infer, _ = infer_pd_inference(dataloader)

    # assert (order_static[:7] == order_pd_infer[:7]).all()

    return order_static[:k].tolist()


@run_once
def get_bm25_model(style="poem/*/*.txt"):

    title_writer_list = []
    poems_list = []
    for file in glob.glob(style):
        # print(file)
        with open(file, "r") as f:
            poem_content = f.read().strip()
        poems_list.append(poem_content)
        title_writer_list.append(poem_content.strip().split("\n")[:2])
        title_writer_list[-1].append(file)

    contents_words = [jieba.lcut(x) for x in poems_list]
    bm25 = BM25Okapi(contents_words)
    print("BM25 model build done.")
    return title_writer_list, bm25


if __name__ == "__main__":
    poems_list, bm25 = get_bm25_model()
    bm25_pred = []
    for sentence in [
        "这首诗, 惟草木之零落兮，恐美人之迟暮",
        "零丁洋里叹零丁",
        "青鸟殷勤为探看",
        "泊船瓜洲",
        "五十弦翻塞外声”是哪位诗人写的？这句诗的后半句是什么？",
        "其中“八百里”指什么",
    ]:

        sentence_words = jieba.lcut(sentence)
        doc_scores = bm25.get_scores(sentence_words)
        max_score_page_idx = doc_scores.argsort()[-1]
        bm25_pred.append(poems_list[max_score_page_idx])

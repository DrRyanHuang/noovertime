import paddle
from paddlenlp.transformers import AutoModel, AutoTokenizer
from model import inference, get_model_tokenizer
from dataset import get_inference_dataloader, get_infer_dataset


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
    order, _ = inference(model, dataloader)

    return order[:k].tolist()

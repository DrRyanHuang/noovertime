import paddle
from paddlenlp.transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModel.from_pretrained("model")


def get_topk(q, a, k):
    model.eval()

    with paddle.no_grad():
        q_inputs = tokenizer(text=q, max_seq_len=512, padding="max_length")
        query_input_ids = paddle.to_tensor([q_inputs["input_ids"]])
        query_token_type_ids = paddle.to_tensor([q_inputs["token_type_ids"]])
        _, cls_embedding = model(query_input_ids, query_token_type_ids)

        a_inputs = tokenizer(text=a, max_seq_len=512, padding="max_length")
        answer_input_ids = paddle.to_tensor(a_inputs["input_ids"])
        answer_token_type_ids = paddle.to_tensor(a_inputs["token_type_ids"])
        _, cls_embedding = model(answer_input_ids, answer_token_type_ids)

        cos_sim = paddle.dot(cls_embedding, cls_embedding) / (
            paddle.linalg.norm(cls_embedding) * paddle.linalg.norm(cls_embedding)
        )

    values, indices = paddle.topk(cos_sim, k=k)

    return indices.numpy().tolist()

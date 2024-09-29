import paddle
from model import get_model_tokenizer

paddle.seed(1107)
MODEL_ROOT = "ernie_ckpt"
model, tokenizer = get_model_tokenizer("ernie_ckpt")


model.eval()
input_spec = [
    paddle.static.InputSpec(shape=[None, 128], dtype="int64"),  # input_ids
    paddle.static.InputSpec(shape=[None, 128], dtype="int64"),  # segment_ids
]

# 3.调用 paddle.jit.save 接口转为静态图模型
path = "ernie_ckpt_static"

paddle.jit.save(layer=model, path=path, input_spec=input_spec)

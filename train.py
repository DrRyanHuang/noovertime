import os
import time

import paddle
import paddle.nn.functional as F


from dataset import get_dataloader
from model import get_model_tokenizer, evaluate, inference


def train(
    model,
    tokenizer,
    train_data_loader,
    save_dir="ernie_ckpt",
    epochs=1,
    save_or_not=True,
):
    """
    ernie_ckpt: 训练过程中保存模型参数的文件夹
    epochs: 训练轮次
    """

    optimizer = paddle.optimizer.AdamW(
        learning_rate=5e-5, parameters=model.parameters()
    )
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    best_acc = 0
    global_step = 0  # 迭代次数

    tic_train = time.time()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = (
                batch["input_ids"],
                batch["token_type_ids"],
                batch["labels"],
            )

            # 计算模型输出、损失函数值、分类概率值、准确率
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            # 每迭代10次，打印损失函数值、准确率、计算速度
            global_step += 1
            if global_step % 10 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (
                        global_step,
                        epoch,
                        step,
                        loss,
                        acc,
                        10 / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()

            # 反向梯度回传，更新参数
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        print("global step", global_step, end=" ")

        # 每完成一个 epoch，评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
        acc_eval = evaluate(model, criterion, metric, dev_data_loader)
        if acc_eval > best_acc and save_or_not:
            best_acc = acc_eval
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    return model, tokenizer


if __name__ == "__main__":

    model_path = "ernie_ckpt"
    # model_path = "ernie-3.0-xbase-zh"
    data_path = "combine.json"

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model, tokenizer = get_model_tokenizer(model_path)
    train_data_loader, dev_data_loader = get_dataloader(
        data_path, tokenizer, train_bs=32, dev_bs=128
    )
    model, tokenizer = train(model, tokenizer, train_data_loader, epochs=1)

    inference(model, dev_data_loader)

# -----------------------------------------------------------------------------------
# test_ds = None

# # 加载ERNIR 3.0最佳模型参数
# params_path = "ernie_ckpt/model_state.pdparams"
# state_dict = paddle.load(params_path)
# model.set_dict(state_dict)

# # 也可以选择加载预先训练好的模型参数结果查看模型训练结果
# # model.set_dict(paddle.load('ernie_ckpt_trained/model_state.pdparams'))

# print("ERNIE 3.0 在lcqmc的dev集表现", end=" ")
# eval_acc = evaluate(model, criterion, metric, dev_data_loader)


# -----------------------------------------------------------------------------------
# # 语义匹配结果预测与保存, 加载微调好的模型参数进行语义匹配预测，并保存预测结果


# # 测试集数据预处理，利用分词器将文本转化为整数序列
# trans_func_test = functools.partial(
#     preprocess_function, tokenizer=tokenizer, max_seq_length=128, is_test=True
# )
# test_ds_trans = test_ds.map(trans_func_test)

# # 进行采样组batch
# collate_fn_test = DataCollatorWithPadding(tokenizer)
# test_batch_sampler = BatchSampler(test_ds_trans, batch_size=32, shuffle=False)
# test_data_loader = DataLoader(
#     dataset=test_ds_trans, batch_sampler=test_batch_sampler, collate_fn=collate_fn_test
# )


# # 模型预测分类结果
# label_map = {0: "不相似", 1: "相似"}
# results = []
# model.eval()
# for batch in test_data_loader:
#     input_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
#     logits = model(batch["input_ids"], batch["token_type_ids"])
#     probs = F.softmax(logits, axis=-1)
#     idx = paddle.argmax(probs, axis=1).numpy()
#     idx = idx.tolist()
#     preds = [label_map[i] for i in idx]
#     results.extend(preds)


# # 存储LCQMC预测结果
# test_ds = load_dataset("lcqmc", splits=["test"])
# res_dir = "./results"
# if not os.path.exists(res_dir):
#     os.makedirs(res_dir)
# with open(os.path.join(res_dir, "lcqmc.tsv"), "w", encoding="utf8") as f:
#     f.write("label\tquery\ttitle\n")
#     for i, pred in enumerate(results):
#         f.write(pred + "\t" + test_ds[i]["query"] + "\t" + test_ds[i]["title"] + "\n")

import argparse
import numpy as np

import paddle
import paddle.inference as paddle_infer
from paddle.inference import PrecisionType

from utils import run_once, DotDict


@run_once
def init_predictor(args):
    if args.model_dir != "":
        config = paddle_infer.Config(args.model_dir)
    else:
        config = paddle_infer.Config(args.model_file, args.params_file)

    config.enable_memory_optim()

    gpu_precision = PrecisionType.Float32
    if args.run_mode == "gpu_fp16":
        gpu_precision = PrecisionType.Half

    config.enable_use_gpu(1000, 0, gpu_precision)

    if args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Int8,
            use_static=False,
            use_calib_mode=True,
        )
    if args.use_dynamic_shape and args.use_collect_shape:
        config.collect_shape_range_info(args.dynamic_shape_file)
    elif args.use_dynamic_shape and not args.use_collect_shape:
        config.enable_tuned_tensorrt_dynamic_shape(args.dynamic_shape_file)
    predictor = paddle_infer.create_predictor(config)
    return predictor


def parse_args_cpu_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./ernie_ckpt_static/inference.pdmodel",
        help="model filename",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./ernie_ckpt_static/inference.pdiparams",
        help="parameter filename",
    )
    parser.add_argument("--batch_size", type=int, default=49, help="batch size")
    return parser.parse_args()


def main_cpu_demo():
    args = parse_args_cpu_demo()

    # 创建 config
    config = paddle_infer.Config(args.model_file, args.params_file)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    # input_names = predictor.get_input_names()
    input_ids_handle = predictor.get_input_handle("input_ids")
    token_type_ids_handle = predictor.get_input_handle("token_type_ids")

    # 设置输入
    input_ids = np.random.randn(args.batch_size, 128).astype("int64")
    input_ids_handle.reshape([args.batch_size, 128])
    input_ids_handle.copy_from_cpu(input_ids)

    token_type_ids = np.random.randint(0, high=2, size=(args.batch_size, 128)).astype(
        "int64"
    )
    token_type_ids_handle.reshape([args.batch_size, 128])
    token_type_ids_handle.copy_from_cpu(token_type_ids)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./ernie_ckpt_static/inference.pdmodel",
        help="Model filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./ernie_ckpt_static/inference.pdiparams",
        help="Parameter filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )
    parser.add_argument(
        "--use_dynamic_shape",
        type=int,
        default=0,
        help="Whether use trt dynamic shape.",
    )
    parser.add_argument(
        "--use_collect_shape",
        type=int,
        default=0,
        help="Whether use trt collect shape.",
    )
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="The file path of dynamic shape info.",
    )

    return parser.parse_args()


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i])

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def main_gpu_demo():

    args = parse_args()
    pred = init_predictor(args)

    input_ids = np.random.randint(0, high=20, size=(49, 128)).astype("int64")
    token_type_ids = np.random.randint(0, high=2, size=(49, 128)).astype("int64")

    result = run(pred, [input_ids, token_type_ids])
    print("class index: ", np.argmax(result[0][0]))


@run_once
def get_default_predictor_config():

    return DotDict(
        {
            "dynamic_shape_file": "",
            "model_dir": "",
            "model_file": "./ernie_ckpt_static/inference.pdmodel",
            "params_file": "./ernie_ckpt_static/inference.pdiparams",
            "run_mode": "gpu_fp16",
            "use_collect_shape": 0,
            "use_dynamic_shape": 0,
        }
    )


def run_infer(inputs, predictor=None, args=None):
    # https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/gpu/resnet50

    if predictor is None:
        if args is None:
            args = get_default_predictor_config()
        predictor = init_predictor(args)

    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(inputs[i].shape)
        input_tensor.copy_from_cpu(inputs[i])

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


if __name__ == "__main__":
    if paddle.device.is_compiled_with_cuda():
        main_gpu_demo()
    else:
        main_cpu_demo()

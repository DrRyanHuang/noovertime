# 请选手按照以下命令格式执行推理脚本

export PYTHON_ROOT=/home/aistudio/.conda/envs/paddlepaddle-env
export PATH=$PYTHON_ROOT/bin:/home/opt/cuda_tools:$PATH
export LD_LIBRARY_PATH=$PYTHON_ROOT/lib:/home/opt/nvidia_lib:$LD_LIBRARY_PATH

unset PYTHONHOME
unset PYTHONPATH

$PADDLEPADDLE_PYTHON_PATH -m pip install redis qianfan jieba

# AK和SK请自己注册获取
$PADDLEPADDLE_PYTHON_PATH inference.py --AK="" \
                       --SK="" \
                       --test_path="dataset.json" \
                       --api_path="api_list.json" \
                       --save_path="result.json"
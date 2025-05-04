set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
# export RAY_DEBUG=1

MODEL_PATH="Meta-Llama-3.1-8B-Instruct"
DIR_NAME=self_chat/baseline
python3 -m verl.trainer.infer \
    model.path=$MODEL_PATH \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_same_model=True \
    exp.test_baseline=True \
    exp.exp_name=llama3.1_8b_baseline $@ 2>&1 | tee ./llama3.1_8b_baseline.log
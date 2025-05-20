set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export RAY_DEBUG=1

# The following are examples of how to run the code for different models and configurations.
# !!!! For inference, the temperature should be set to 0.7
# !!!! For evaluation, temperature: 0.0 top_p: 1 frequency_penalty: 0 presence_penalty: 0

# 1. Self-chat with vllm offline infer (Meta-Llama-3.1-8B-Instruct)
MODEL_PATH="ckpt of Meta-Llama-3.1-8B-Instruct"
DIR_NAME=self_chat/baseline
python3 -m verl.trainer.infer \
    model.path=$MODEL_PATH \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_same_model=True \
    exp.test_baseline=True \
    exp.exp_name=self_chat_llama3.1_8b_baseline $@ 2>&1 | tee ./self_chat_llama3.1_8b_baseline.log

# 2. GPT-4o-As-Partner with vllm offline infer (Meta-Llama-3.1-8B-Instruct)
MODEL_PATH="ckpt of Meta-Llama-3.1-8B-Instruct"
DIR_NAME=4o_partner/baseline
python3 -m verl.trainer.infer \
    model.path=$MODEL_PATH \
    exp.opponent_model=../config/gpt4o.yaml \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_baseline=True \
    exp.exp_name=4o_partner_llama3.1_8b_baseline $@ 2>&1 | tee ./4o_partner_llama3.1_8b_baseline.log

# 3. Self-chat with vllm offline infer (BC, GRPO, AMPO)
MODEL_PATH="ckpt of BC, GRPO, AMPO"
DIR_NAME=self_chat/AML
python3 -m verl.trainer.infer \
    model.path=$MODEL_PATH \
    exp.opponent_model=../config/gpt4o.yaml \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_thinking=True \
    exp.test_same_model=True \
    exp.exp_name=self_chat_llama3.1_8b_AML $@ 2>&1 | tee ./result/log/self_chat_llama3.1_8b_AML.log


# 4. GPT-4o-As-Partner with vllm offline infer (BC, GRPO, AMPO)
MODEL_PATH="ckpt of BC, GRPO, AMPO"
DIR_NAME=4o_partner/AML
python3 -m verl.trainer.infer \
    model.path=$MODEL_PATH \
    exp.opponent_model=../config/gpt4o.yaml \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_thinking=True \
    exp.exp_name=4o_partner_llama3.1_8b_AML $@ 2>&1 | tee ./result/log/4o_partner_llama3.1_8b_AML.log

# 5. Self-chat with api LRMs
MODEL=../config/qwq.yaml
DIR_NAME=self_chat/baseline
python3 -m verl.trainer.infer \
    exp.model_name=$MODEL \
    exp.api=True \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_same_model=True \
    exp.test_baseline=True \
    exp.test_thinking=True \
    exp.model_process_num=5 \
    exp.exp_name=self_chat_qwq $@ 2>&1 | tee ./result/log/self_chat_qwq.log

# 6. GPT-4o-As-Partner with api LRMs
MODEL=../config/qwq.yaml
DIR_NAME=4o_partner/baseline
python3 -m verl.trainer.infer \
    exp.model_name=$MODEL \
    exp.api=True \
    exp.opponent_model=../config/gpt4o.yaml \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_thinking=True \
    exp.test_baseline=True \
    exp.model_process_num=5 \
    exp.exp_name=4o_partner_qwq $@ 2>&1 | tee ./result/log/4o_partner_qwq.log

# 7. Self-chat with api LLMs
MODEL=../config/gpt4o.yaml
DIR_NAME=self_chat/baseline
python3 -m verl.trainer.infer \
    exp.model_name=$MODEL \
    exp.api=True \
    exp.dir_name=$DIR_NAME \
    exp.env=sotopia \
    exp.test_same_model=True \
    exp.test_baseline=True \
    exp.model_process_num=5 \
    exp.exp_name=self_chat_gpt4o $@ 2>&1 | tee ./result/log/self_chat_gpt4o.log

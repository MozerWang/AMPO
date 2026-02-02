# Adaptive Social Learning via Mode Policy Optimization for Language Agentss

<p align="center">
    <img src="src/aml.png" width="70%" height="50%">
</p>

<div align="center">
<br>
<a href="https://scholar.google.com.hk/citations?user=glV21ZsAAAAJ&hl=zh-CN">Minzheng Wang</a><sup><span>1,2</span></sup>, 
<a>Yongbin Li</a><sup><span>3</span></sup>,
<a>Haobo Wang</a><sup><span>4</span></sup>,
<a href="https://xinghuazhang.top/">Xinghua Zhang</a><sup><span>3🌟</span></sup>,
<br>
<a>Nan Xu</a><sup><span>1</span></sup>,
<a>Bingli Wu</a><sup><span>3</span></sup>,
<a>Fei Huang</a><sup><span>3</span></sup>,
<a>Haiyang Yu</a><sup><span>3</span></sup>,
<a>Wenji Mao</a><sup><span>1,2🌟</span></sup>
<br>

🌟 Corresponding author

<sup>1</sup> MAIS, Institute of Automation, Chinese Academy of Sciences<br>
<sup>2</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
<sup>3</sup> Tongyi Lab, Alibaba Group<br>
<sup>4</sup> Peking University<br>

<font size=3><div align='center' >  [[📖 ArXiv Paper](https://arxiv.org/pdf/2505.02156)] [[📊 Code](https://github.com/MozerWang/AMPO)] [[😊 Data](https://huggingface.co/datasets/iiiiwis/AMPO)] </div></font>

</div>


## 👀 Overview
This repository contains code and data for our paper **Adaptive Social Learning via Mode Policy Optimization for Language Agents**. In this paper, we propose the **A**daptive **M**ode **L**earning framework (**AML**) to empower social agents with the capability for adaptive thinking, enabling them to effectively respond in accordance with the dynamics of social interaction context.
Specifically, we first develop four thinking modes inspired by hierarchical cognitive control theory, covering a spectrum from intuitive response, through shallow and strategic thinking, to deep deliberation. 
Next, we perform the injection of thinking modes, which consists of behavioral cloning for learning basic modes and RL-based adaptive thinking mode enhancement.
For RL-based enhancement, we contrapuntally develop the **A**daptive **M**ode **P**olicy **O**ptimization (**AMPO**) algorithm, which incorporates the mode-level and sample-level information into advantage estimation to strengthen the context-aware thinking mode switching.
In terms of reward, we design three types of reward functions, including answer reward, format reward, and answer length reward, providing feedback for choosing the appropriate thinking mode and answer.

## Main Results
<p align="center">
    <img src="./src/exp1.png" width="70%" height="50%">
</p>
<p align="center">
    <img src="./src/exp2.png" width="70%" height="50%">
</p>

> Extensive experimental results show that AML and AMPO achieve the SOTA performances in comparison with strong baselines. Details can be found in the paper.

## 🔥 Update

- [2026.01.25]🙌AMPO is accepted to the ICLR 2026, congratulations to all co-authors!
- [2025.05.04]🔥AMPO is coming! We have released the [paper](https://arxiv.org/pdf/2505.02156), [code](https://github.com/MozerWang/AMPO), [data](https://huggingface.co/datasets/iiiiwis/AMPO)! Feel free to contact me via email if you have any issues!

## 🔧How to use
<p align="center">
    <img src="./src/alg.png" width="70%" height="50%">
</p>

> The full optimization procedure. We employ a two-phase training procedure: The first phase utilizes mode behavioral cloning to enable the model to understand and follow specific thinking modes accurately. In the second phase, we perform adaptive mode policy optimization to enhance the adaptive thinking mode switch and reasoning.

**Step1** Create conda environment and Install other dependencies.
1. Clone this repository
```shell
git clone https://github.com/MozerWang/AMPO
cd AMPO
```
2. Create BC conda environment (LLaMA Factory).
```shell
conda create --name BC python=3.11 -y
conda activate BC
cd BC 
pip install -e ".[torch,metrics]"
```
3. Create RL conda environment (verl).
```shell
# RL environment (verl)
conda create --name RL python=3.11 -y
conda activate RL
cd RL
pip3 install -e .[vllm]
pip install -r requirements.txt
```

> *you can also refer to the install instruction in [verl](https://github.com/volcengine/verl) and [llamafactory](https://github.com/hiyouga/LLaMA-Factory/).*

**Step2** Download the training data from huggingface
```shell
git lfs install
git clone https://huggingface.co/datasets/iiiiwis/AMPO
```
**Step3** Preparing the Model API

1. (**Must**) Set up your OPENAI key in config/gpt_4o.yaml (Evaluation)
```shell
api_key: "Your OPENAI key"
api_url: "API URL"
```

2. (**Must**) Set up your key in config/qwen2.5_72b_instruct.yaml (Reward Model)
```shell
api_key: "Your key"
api_url: "API URL"
# We also recommend using vLLM. And we use HTTP server that implements OpenAI’s Completions and Chat API.
# Set up your vLLM settings in config/*.yaml
```
**Step4** Behavior Cloning Training
```shell
conda activate BC
cd BC
## (Must) Firstly set the bc_training_data_path in ./BC/data/dataset_info.yaml
sh train.sh
```

**Step5** RL Training
```shell
conda activate RL
cd RL
## (Must) Firstly, translate the rl training data into ".parquet" format by using the script in ./RL/example/data_preprocess/sotopia.py
sh sotopia_ampo_llama3.1_8b.sh
sh sotopia_ampo_qwen2.5_7b.sh
```

**Step6** Evaluation and Inference
```shell
conda activate RL
cd RL
sh infer.sh
## show result
python result.py --env sotopia --data_path your_result_path
```

## Acknowledgement
Thanks for these amazing work!
- [verl](https://github.com/volcengine/verl)
- [vllm](https://github.com/vllm-project/vllm)
- [llamafactory](https://github.com/hiyouga/LLaMA-Factory/)
- [dat](https://github.com/likenneth/dialogue_action_token)
- [sotopia](https://github.com/sotopia-lab/sotopia)

## Citation
```
@inproceedings{
wang2025ampo,
title={Adaptive Social Learning via Mode Policy Optimization for Language Agents},
author={Minzheng Wang and Yongbin Li and Haobo Wang and Xinghua Zhang and Nan Xu and Bingli Wu and Fei Huang and Haiyang Yu and Wenji Mao},
booktitle={Proceedings of ICLR},
year={2026},
url={https://openreview.net/forum?id=GG7YQnsdhp}
}
```

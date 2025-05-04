# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.insert(0, './')
import argparse
import os
import json
import sys
# print("Current working directory:", os.getcwd())
# print("Python path:", sys.path)
# print("File location:", __file__)

import pandas as pd
from datasets import Dataset, Features, Value, Sequence

from tqdm.auto import tqdm
from verl.prompt import auto_thinking_prompt

features = Features({
    'dialog': Sequence(Value('string')),
    'env_id': Value('int64'),
    'turn': Value('int64'),
    'instruction': Value('string'),
    'agent1_name': Value('string'),
    'agent2_name': Value('string'),
    'agent1_goal': Value('string'),
    'agent2_goal': Value('string'),
    'rollout': Value('int64'),
    'state': {
        'agent1': {
            'score': Value('float32')
        },
        'agent2': {
            'score': Value('float32')
        },
        'step': Value('int64')
    },
    'agent1_goal_score': Value('float64'),
    'agent2_goal_score': Value('float64'),
    'agent1_avg_score': Value('float64'),
    'agent2_avg_score': Value('float64')
})


def generate_rl_dataset(train_input_data, local_dir='~/data/demo/'):
    with open(train_input_data, 'r') as f:
        datas = json.load(f)

    train_dataset = Dataset.from_list(datas, features=features)
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = example.pop('instruction')
            env_id = example.pop('env_id')
            turn = example.pop('turn')
            dialog = example.pop('dialog')
            agent1_name = example.pop('agent1_name')
            agent2_name = example.pop('agent2_name')
            agent1_goal = example.pop('agent1_goal')
            agent2_goal = example.pop('agent2_goal')
            state_reward = example.pop('state')
            rollout = example.pop('rollout')
            assert 0 <= state_reward['agent1']['score'] <= 10, "reward should in [0, 10]"
            assert 0 <= state_reward['agent2']['score'] <= 10, "reward should in [0, 10]"
            
            data = {
                "data_source": "demo",
                "prompt": [{
                    "role": "system", 
                    "content": auto_thinking_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                "ability": "reason",
                "reward_model": {
                    "style": "model",
                    "turn": turn,
                    "dialog": dialog,
                    "agent1_name": agent1_name,
                    "agent2_name": agent2_name,
                    "agent1_goal": agent1_goal,
                    "agent2_goal": agent2_goal,
                    "state_reward": state_reward,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'env_id': env_id,
                    'turn': turn,
                    'rollout': rollout
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, 'train_0323_2330.parquet')
    train_dataset.to_parquet(local_path)

    print(f"Train dataset saved to {local_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_input_data', type=str, default='/mnt/workspace/minzheng_wang/SlowThinking/src/infer/output/dpo_data/0221/llama3.1_8b/split_seed=none_batch=600_temp=0.7_llama3.1_8b_auto_dpo_0305.json')
    # parser.add_argument('--local_dir', type=str, default='./data/sotopia')
    # parser.add_argument('--model_data', type=str, default='llama3.1_8b')

    # parser.add_argument('--train_input_data', type=str, default='/mnt/workspace/minzheng_wang/SlowThinking/src/infer/output/dpo_data/0221/qwen2_7b/split_seed=none_batch=600_temp=0.7_qwen2_7b_auto_dpo_0305.json')
    # parser.add_argument('--local_dir', type=str, default='./data/sotopia')
    # parser.add_argument('--model_data', type=str, default='qwen2_7b')

    parser.add_argument('--train_input_data', type=str, default='/mnt/workspace/minzheng_wang/SlowThinking/src/infer/output/demo/self_chat/split_seed=none_batch=750_temp=1.0_qwen2.5_7b_auto_buffer_0323_2330.json')
    parser.add_argument('--local_dir', type=str, default='./data/demo')
    parser.add_argument('--model_data', type=str, default='qwen2.5_7b')
    
    
    args = parser.parse_args()

    generate_rl_dataset(args.train_input_data, os.path.join(args.local_dir, args.model_data))
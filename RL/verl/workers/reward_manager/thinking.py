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

from verl import DataProto
from verl.utils.reward_score import think_compute_score, baseline_compute_score
from verl.utils.yaml_load import load
import json
import torch

reward_llm = ['../config/qwen2.5_72b_instruct.yaml']

class ThinkingRewardManager:
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, test_baseline=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        if test_baseline:
            self.compute_score = baseline_compute_score
        else:
            self.compute_score = compute_score or think_compute_score
        self.reward_models = []
        for llm in reward_llm:
            reward_model = load(open(llm, 'r'))
            self.reward_models.append(reward_model)
        print(json.dumps(self.reward_models, indent=4))

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        length_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        level_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        reward_batch = []
        valid_response_length_batch = []
        data_source_batch = []
        sequences_str_batch = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_length_batch.append(valid_response_length)
            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # sequences_str = self.tokenizer.decode(sequences)
            
            output_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            state_infor = data_item.non_tensor_batch['reward_model']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            reward_batch.append({"state_infor":state_infor,
                                 "data_source":data_source,
                                 "extra_info":extra_info,
                                 "result":output_text,
                                 "used_token":len(valid_response_ids)})
            data_source_batch.append(data_source)
            sequences_str_batch.append(output_text)

        scores = self.compute_score(
            reward_batch, self.reward_models, self.tokenizer
        )

        for i, reward_infor in enumerate(scores):
            score = reward_infor['score']
            level = float(reward_infor['level'])
            used_token = float(reward_infor['used_token'])
            reward_tensor[i, valid_response_length_batch[i] - 1] = score
            length_tensor[i, valid_response_length_batch[i] - 1] = used_token
            level_tensor[i, valid_response_length_batch[i] - 1] = level
            if data_source_batch[i] not in already_print_data_sources:
                already_print_data_sources[data_source_batch[i]] = 0

            if already_print_data_sources[data_source_batch[i]] < self.num_examine:
                already_print_data_sources[data_source_batch[i]] += 1
                print(sequences_str_batch[i])

            print(f" Rollout ".center(80, '-'))
            print(f"{sequences_str_batch[i]}")
            print("\n" + "-"*80)
            print(f" Final Score ".center(80, '-'))
            print(f"Score: {score}")
            print(json.dumps(reward_infor, indent=4))
            print("="*80 + "\n")

        return reward_tensor, length_tensor, level_tensor, scores

import torch
from typing import Literal, List
from langchain.output_parsers import PydanticOutputParser
import os
import re
import json
import numpy as np
from verl.envs.sotopia_utils.utils import ScriptBackground, EnvResponse, get_bio, format_bad_output, prefix_prompt, additional_instruction
from verl.envs.utils.generate import api_generate
from verl.envs.utils.utils import load
from datasets import Dataset
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

current_file_path = os.path.abspath(__file__)
directory = os.path.dirname(current_file_path)

class BatchSotopiaEnv():
    def __init__(self, model_name: None,
                tokenizer: None,
                env_model: None,
                opponent_model: None,
                test_same_model: bool = False,
                max_turns: int = 20,
                temperature: float = 0.7,
                batch_size: int = 8,
                actor_role: Literal['agent1', 'agent2'] = 'agent1',
                system_prompt: str = "",
                saving_path: str = None,
                api: bool = False,
                port: int = 8000,
                test_thinking: bool = False,
                test_baseline: bool = False,
                model_process_num: int = 20,
                eval_process_num: int = 15,
                save_folder: str = "",
                test_hard_only: bool = False,
                total_episodes: int = 450,
                rollout_prompt_length: int = 6144,
                ):
        self.batch_size = batch_size
        self.model_process_num = model_process_num
        self.eval_process_num = eval_process_num
        self.env_model = env_model
        self.max_turns = max_turns
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.api = api
        self.port = port
        self.test_thinking = test_thinking
        self.test_baseline = test_baseline
        self.saving_path = saving_path
        self.save_folder = save_folder
        self.actor_role = actor_role 
        self.test_hard_only = test_hard_only
        self.additional_instruction = additional_instruction
        self.total_episodes = total_episodes
        self.rollout_prompt_length = rollout_prompt_length
        self.load_scenarios()
        
        self.test_same_model = test_same_model
        if self.api:
            if "," in model_name:
                self.model = model_name.split(',')
            else:
                self.model = [model_name]      
        else:
            self.model = model_name
            
        
        if opponent_model == None:
            self.reuse_model = True
        else:
            self.reuse_model = False
            if "," in opponent_model:
                self.opponent_model = opponent_model.split(',')
            else:
                self.opponent_model = [opponent_model]  
            
        self.tokenizer = tokenizer

        # four model chat ways: 
        # 1. self-chat (same model) (450 episodes)
        # 2. vllm offline infer with same vllm base model (both vllm offline infer eg. qwen-lora vs qwen 900 episodes)
        # 3. vllm offline infer with api opponent model (vllm offline infer vs api eg. qwen-lora vs gpt4o 900 episodes)
        # 4. api model1 vs api model2 (900 episodes)
        
        if test_same_model:
            assert opponent_model == None, "way1: opponent_model must be empty when test_same_model is True"
        elif opponent_model:
            assert not test_same_model and self.api, "way3,4: With opponent_model, test_same_model must be False and api must be True"
        else:  # opponent_model is empty and not test_same_model
            assert not self.api, "way2: When opponent_model is empty (not test_same_model), api must be False"

    def load_scenarios(self):
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/env_agent_combos.json'), 'r') as f:
            self.env_agent_combos = json.load(f)
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/envs.json'), 'r') as f:
            self.envs = json.load(f)
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/agents.json'), 'r') as f:
            self.agent_profiles = json.load(f)
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/sotopia_hard.json'), 'r') as f:
            self.hard_ids = json.load(f)
    
    def get_current_prompt(self, env_idx: int):
        # accumulate the current dialog into a prompt
        prompt = "\n" + "\n".join(self.cur_dialogs[env_idx])
        prompt = self.agent1_intro[env_idx].format(turn=self.cur_turns[env_idx]) + prompt if self.cur_turns[env_idx] % 2 == 0 else self.agent2_intro[env_idx].format(turn = self.cur_turns[env_idx]) + prompt
        return prompt
    
    def _clean_tags(self, text):
        # use regex to remove the tags in the text
        return re.sub(r'<.*?>', '', text)
    
    def _judge_terminate(self) -> bool:
        for i in range(self.batch_size):
            if self.terminations[i]:
                continue   
            # 1. exceeds the max_turns
            if self.cur_turns[i] >= self.max_turns:
                self.terminations[i] = True
                print(f'{self.env_ids[i]} finished at turn {self.cur_turns[i] - 1}')
            elif any([_ in self.cur_dialogs[i][-1].lower() for _ in ["left the conversation", "leave the conversation"]]):
                self.terminations[i] = True
                print(f'{self.env_ids[i]} finished at turn {self.cur_turns[i] - 1}')
            else:
                self.terminations[i] = False
        done = all(self.terminations)
        return done

    def reset(self, env_ids: List[int] = None, actor_role: str = "agent1", restore: bool = False):
        batch_size = len(env_ids)
        self.processing_turn  = 0
        self.batch_size = batch_size
        self.actor_role = actor_role
        # 初始化批量环境的状态
        self.cur_turns = [0] * batch_size  # 每个环境的当前回合
        self.cur_dialogs = [[] for _ in range(batch_size)]  # 每个环境的对话历史
        self.cur_rewards = [[] for _ in range(batch_size)]  # 每个环境的奖励
        self.cur_fullout = [[] for _ in range(batch_size)]  # 每个环境的思考过程
        self.terminations = [False] * batch_size  # 每个环境是否终止
        self.early_failed = [True] * batch_size
        self.agent1_prefix = [''] * batch_size
        self.agent2_prefix = [''] * batch_size
        self.agent1_intro = [''] * batch_size 
        self.agent2_intro = [''] * batch_size
        self.complete_intro = [''] * batch_size
        self.agent1_name = [''] * batch_size
        self.agent2_name = [''] * batch_size
        self.agent1_goal = [''] * batch_size
        self.agent2_goal = [''] * batch_size
        self.p1_scores = [{}] * batch_size
        self.p2_scores = [{}] * batch_size
        self.env_ids = [None] * batch_size
        self.env_id_start = env_ids[0] if self.actor_role == 'agent1' else env_ids[0] + self.total_episodes
        # initialize the sotopia environment
        for i in range(self.batch_size):
            env_id = env_ids[i]
            self.env_ids[i] = env_id
            if self.test_hard_only and env_id not in self.hard_ids:
                self.terminations[i] = True
                self.cur_rewards[i].append(-100)
                
            env_agent_combo_storage = self.env_agent_combos[env_id]
            env_profile = self.envs[env_agent_combo_storage['env_id']]
            agent_ids = env_agent_combo_storage['agent_ids']
            agent_profiles = [self.agent_profiles[id] for id in agent_ids]

            self.agent1_name[i] = agent_profiles[0]['first_name'] + " " + agent_profiles[0]['last_name']
            self.agent2_name[i] = agent_profiles[1]['first_name'] + " " + agent_profiles[1]['last_name']
            self.agent1_prefix[i] = prefix_prompt.format(agent = self.agent1_name[i])
            self.agent2_prefix[i] = prefix_prompt.format(agent = self.agent2_name[i])
            self.agent1_goal[i] = env_profile['agent_goals'][0]
            self.agent2_goal[i] = env_profile['agent_goals'][1]
            
            all_background = ScriptBackground(
                scenario=env_profile['scenario'],
                p1_background=get_bio(env_profile['relationship'], agent_profiles[0], agent_id=0),
                p2_background=get_bio(env_profile['relationship'], agent_profiles[1], agent_id=1),
                p1_goal=f"{env_profile['agent_goals'][0]}",
                p2_goal=f"{env_profile['agent_goals'][1]}",
                p1_name=self.agent1_name[i],
                p2_name=self.agent2_name[i],
            )
            agent1_background = ScriptBackground(
                scenario=env_profile['scenario'],
                p1_background=get_bio(env_profile['relationship'], agent_profiles[0], agent_id=0),
                p2_background=get_bio(env_profile['relationship'], agent_profiles[1], agent_id=1),
                p1_goal=f"{env_profile['agent_goals'][0]}",
                p2_goal="Unknown",
                p1_name=self.agent1_name[i],
                p2_name=self.agent2_name[i],
            ).to_natural_language()
            
            agent2_background = ScriptBackground(
                scenario=env_profile['scenario'],
                p1_background=get_bio(env_profile['relationship'], agent_profiles[0], agent_id=0),
                p2_background=get_bio(env_profile['relationship'], agent_profiles[1], agent_id=1),
                p1_goal="Unknown",
                p2_goal=f"{env_profile['agent_goals'][1]}",
                p1_name=self.agent1_name[i],
                p2_name=self.agent2_name[i],
            ).to_natural_language()
            
            self.agent1_intro[i] = agent1_background.strip() + '\n' + self.additional_instruction
            self.agent2_intro[i] = agent2_background.strip() + '\n' + self.additional_instruction
            
            self.complete_intro[i] = self._clean_tags(all_background.to_natural_language())
            self.agent1_intro[i] = self.agent1_prefix[i] + self._clean_tags(self.agent1_intro[i])
            self.agent2_intro[i] = self.agent2_prefix[i] + self._clean_tags(self.agent2_intro[i])

            # initialize the dialog
            self.cur_dialogs[i].append(f"Turn {self.cur_turns[i]}: {self.agent1_name[i]} said:")
            self.done = False
        
        if restore:
            with open(self.saving_path, 'r', encoding='utf-8') as fr:
                existing_data = json.load(fr)
 
            for idx in range(batch_size):
                data = existing_data[self.env_id_start + idx]
                self.cur_turns[idx] = len(data['dialog']) - 1 
                self.cur_dialogs[idx] = data['dialog']
                self.cur_rewards[idx] = data['rewards']
                self.terminations[idx] = data['terminals']
                self.p1_scores[idx] = data['agent1_scores']
                self.p2_scores[idx] = data['agent2_scores']
                self.cur_fullout[idx] = data['fullout']
                if self.cur_turns[idx] > self.processing_turn:
                    self.processing_turn = self.cur_turns[idx]
            self.done = self._judge_terminate()
    
    def verl_vllm_process(self, prompt_inputs):
        chat_lst = [chat_lst['prompt'] for chat_lst in prompt_inputs]
        inputs = self.tokenizer.apply_chat_template(chat_lst,
                                        add_generation_prompt=True,
                                        padding=True,
                                        truncation=True,
                                        max_length=self.rollout_prompt_length,
                                        return_tensors='pt',
                                        return_dict=True,
                                        tokenize=True)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        data = DataProto.from_dict(batch_dict)
        test_gen_batch = data.pop(['input_ids', 'attention_mask', 'position_ids'])
        test_gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
        }
        return test_gen_batch
    
    def _act(self, actor: bool = False):
        with torch.no_grad():
            prompt_inputs = []
            infer_list = []
            if self.test_same_model or actor or self.reuse_model:
                if type(self.model) is list:
                    config = [load(open(model, 'r')) for model in self.model]
                else:
                    config = []
                use_api = self.api
            else:
                config = [load(open(model, 'r')) for model in self.opponent_model]
                use_api = True
            
            for cfg in config:
                cfg['run_args']['temperature'] = self.temperature
                
            # collect the samples that need to be inferred
            for env_idx in range(self.batch_size):
                if not self.terminations[env_idx]:
                    temp_dict = {}
                    temp_list = []    
                    if self.system_prompt and (actor or self.test_same_model):
                        temp_dict["system"] = self.system_prompt
                        temp_list.append({"role": "system", "content": self.system_prompt})
                    current_prompt = self.get_current_prompt(env_idx)
                    temp_dict["user"] = current_prompt
                    temp_list.append({"role": "user", "content": current_prompt})
                    tokenized_prompt_len = len(self.tokenizer.apply_chat_template(temp_list, add_generation_prompt=True))
                    if tokenized_prompt_len <= self.rollout_prompt_length:
                        if use_api:
                            prompt_inputs.append({"prompt":temp_dict})
                        else:
                            prompt_inputs.append({"prompt":temp_list})
                        infer_list.append(env_idx)
                    else:
                        self.terminations[env_idx] = True
                        self.early_failed[env_idx] = False
            
            output_texts = []
            if prompt_inputs:
                if use_api:
                    final_results = [None]*len(prompt_inputs)
                    map_ids = list(range(len(prompt_inputs)))
                    while prompt_inputs:
                        results = api_generate(prompt_inputs, config, self.model_process_num, self.port)
                        remove_list = []
                        for idx, result in enumerate(results):
                            if result:
                                final_results[map_ids[idx]] = result
                                remove_list.append(idx)
                        prompt_inputs = [item for idx, item in enumerate(prompt_inputs) if idx not in remove_list]
                        map_ids = [item for idx, item in enumerate(map_ids) if idx not in remove_list]
                    output_texts = final_results
                else:
                    
                    test_gen_batch = self.verl_vllm_process(prompt_inputs)
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.model.world_size)
                    test_output_gen_batch_padded = self.model.generate_sequences(test_gen_batch_padded)
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                    print('validation generation end')
                    output_ids = test_output_gen_batch.batch['responses']
                    output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                for idx, result in enumerate(output_texts):
                    env_idx = infer_list[idx]
                    self.cur_fullout[env_idx].append(result)
                    if self.test_thinking and (actor or self.test_same_model) and not self.test_baseline:
                        if '<|begin_of_answer|>' in result and '<|end_of_answer|>' in result:
                            result = result.split('<|begin_of_answer|>')[-1].split('<|end_of_answer|>')[0].strip()
                        elif '<|begin_of_answer|>' in result:
                            result = result.split('<|begin_of_answer|>')[0]
                            result = result.split('\n\nResponse:\n')[-1].strip()
                        elif '\n\nResponse:\n' in result:    
                            result = result.split('\n\nResponse:\n')[-1].strip()
                        else:
                            result = result.split('\n')[-1].strip()
                    else:
                        if self.test_thinking and (actor or self.test_same_model) and self.test_baseline:
                            result = result[0]
                        if "\"" in result:
                            start = result.index("\"")
                            end = result.rindex("\"")
                            result = result[start:end+1]
                        else:
                            result = result.split('\n')[0].strip()
                    result = "\"" + result.strip("\"") + "\""
                    self.cur_turns[env_idx] += 1
                    self.cur_dialogs[env_idx][-1] += " " + result + '\n'
            print("Finished turns: ", self.processing_turn)
            
    def batch_step(self):
        if not self.done:
            if self.processing_turn % 2 == 0:
                self._act(actor=(self.actor_role == 'agent1'))
            else:
                self._act(actor=(self.actor_role == 'agent2'))
            self.done = self._judge_terminate()
        
        if self.done:
            outputs, ref_list, fail_list = self.get_final_reward()
            rewards = []
            for i, output in enumerate(outputs):
                idx = ref_list[i]
                if idx in fail_list:
                    self.cur_rewards[idx][-1] = output[0]
                else:
                    self.cur_rewards[idx].append(output[0])
                self.p1_scores[idx] = output[1]
                self.p2_scores[idx] = output[2]
            rewards = [cur_reward[-1] for cur_reward in self.cur_rewards if cur_reward[-1] != -100]
            return rewards, self.done
            
        self.processing_turn += 1
        for i in range(self.batch_size):
            if not self.terminations[i]:
                self.cur_dialogs[i].append(f"Turn {self.cur_turns[i]}: {self.agent2_name[i]} said:") if self.cur_turns[i] % 2 != 0 else self.cur_dialogs[i].append(f"Turn {self.cur_turns[i]}: {self.agent1_name[i]} said:")
                self.cur_rewards[i].append(0)

        return 0, self.done
    
    def get_final_reward(self):
        if ',' in self.env_model:
            self.env_model = self.env_model.split(',')
        if type(self.env_model) is list:
            env_llms = [load(open(cfg, 'r')) for cfg in self.env_model]
        else:
            env_llms = load(open(self.env_model, 'r'))
        print("eval model config:")
        print(json.dumps(env_llms, indent=4))
        template = """{history}{schema}"""
        with open(os.path.join(directory, 'sotopia_utils/output_schema.txt'), 'r') as f:
            schema = f.read()
        output_parser = PydanticOutputParser[EnvResponse](pydantic_object=EnvResponse)
        prompt_inputs = []
        outputs = []
        ref_list = []
        fail_list = []
        for env_idx in range(self.batch_size):
            if self.test_hard_only and env_idx not in self.hard_ids:
                continue
            if self.p1_scores[env_idx]:
                continue
            if self.cur_rewards[env_idx] and self.cur_rewards[env_idx][-1] == -100:
                fail_list.append(env_idx)
            ref_list.append(env_idx)
            history = self.complete_intro[env_idx] + "\n".join(self.cur_dialogs[env_idx])
            prompt = template.format(history=history, schema=schema)
            prompt_input = {"prompt": {'user': prompt}}
            prompt_inputs.append(prompt_input)
        output_texts = api_generate(prompt_inputs, env_llms, self.eval_process_num, self.port)

        for env_idx, output_text in enumerate(output_texts):
            try:
                assert output_text!='', "output should not be empty"
                try:
                    parsed_result = output_parser.parse(output_text)
                except Exception as e:
                    reformat_parsed_result = format_bad_output(output_text, format_instructions=output_parser.get_format_instructions(), model_name=env_llms)
                    parsed_result = output_parser.parse(reformat_parsed_result)
                d = parsed_result.agent_1_evaluation.dict() if self.actor_role== 'agent1' else parsed_result.agent_2_evaluation.dict()
                overall_score = sum(d[dimension][1] for dimension in d) / len(d)
                outputs.append((overall_score, parsed_result.agent_1_evaluation.dict(), parsed_result.agent_2_evaluation.dict()))
            except Exception as e:
                outputs.append((-100, {}, {}))
        return outputs, ref_list, fail_list
    
    
    def save_conversation_history(self):
        file_path = self.saving_path
        existing_data  = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as fr:
                existing_data = json.load(fr)
        
        # existing_data = existing_data[:self.env_id_start]
       
        for i in range(self.batch_size):
            tbd = dict(
                env_id=self.env_ids[i],
                complete_intro=self.complete_intro[i],
                dialog=self.cur_dialogs[i],
                rewards=self.cur_rewards[i],
                terminals=self.terminations[i],
                early_failed=self.early_failed[i],
                actor_role=self.actor_role,
                agent1_intro=self.agent1_intro[i],
                agent2_intro=self.agent2_intro[i],
                agent1_name=self.agent1_name[i],
                agent2_name=self.agent2_name[i],
                agent1_goal=self.agent1_goal[i],
                agent2_goal=self.agent2_goal[i],
                agent1_scores=self.p1_scores[i],
                agent2_scores=self.p2_scores[i],
                hard = self.env_ids[i] in self.hard_ids,
                fullout = self.cur_fullout[i]
            )
            try:
                existing_data[self.env_id_start + i] = tbd
            except:
                existing_data.append(tbd)
        with open(file_path, 'w', encoding='utf-8') as fw:
            json.dump(existing_data, fw, indent=4, ensure_ascii=False)
        
        if self.done:
            file = file_path.split('/')[-1]
            raw_directory = "/".join(file_path.split('/')[:-1]) + '/raw'
            if not os.path.exists(raw_directory):
                os.makedirs(raw_directory)
            file_path = os.path.join(raw_directory, file)
            raw_data = []
            for data in existing_data:
                data['rewards'][-1] = -100
                data['agent1_scores'] = {}
                data['agent2_scores'] = {}
                raw_data.append(data)
            with open(file_path, 'w', encoding='utf-8') as fw:
                json.dump(raw_data, fw, ensure_ascii=False, indent=4)
    
    def get_final_result(self):
        with open(self.saving_path, 'r', encoding='utf-8') as fr:
            datas = json.load(fr)
        
        nums, hard_nums = 0, 0
        actor_goal, oppo_goal, goal_avg, hard_actor_goal, hard_oppo_goal, hard_goal_avg  = [], [], [], [], [], []
        
        for data in datas:
            if self.test_hard_only and not data["hard"]:
                continue
            if data['rewards'][-1] == -100 or data['agent1_scores'] == {}:
                continue
            nums += 1
            actor = data['actor_role']
            actor_score = data['agent1_scores'] if actor == 'agent1' else data['agent2_scores']
            oppo_score = data['agent1_scores'] if actor == 'agent2' else data['agent2_scores']
            actor_goal.append(actor_score['goal'][-1])
            oppo_goal.append(oppo_score['goal'][-1])
            goal_avg.append((actor_score['goal'][-1] + oppo_score['goal'][-1]) / 2) 
            if data['hard']:
                hard_nums += 1 
                hard_actor_goal.append(actor_goal[-1])
                hard_oppo_goal.append(oppo_goal[-1])
                hard_goal_avg.append(goal_avg[-1]) 
                    
        print(f"Total Episodes: {nums}")
        print(f"Total Hard Episodes: {hard_nums}")
        if self.test_hard_only:
            metric_dict = {
                "val/hard_actor_goal_mean": np.mean(hard_actor_goal),
                "val/hard_oppo_goal_mean": np.mean(hard_oppo_goal),
                "val/hard_goal_avg_mean": np.mean(hard_goal_avg),
            }       
        else:
            metric_dict = {
                "val/actor_goal_mean": np.mean(actor_goal),
                "val/oppo_goal_mean": np.mean(oppo_goal),
                "val/goal_avg_mean": np.mean(goal_avg),
                "val/hard_actor_goal_mean": np.mean(hard_actor_goal),
                "val/hard_oppo_goal_mean": np.mean(hard_oppo_goal),
                "val/hard_goal_avg_mean": np.mean(hard_goal_avg),
            }     
        for key, value in metric_dict.items():
            print("="*50)
            print(f"{key}: {value:.2f}")
        return metric_dict        
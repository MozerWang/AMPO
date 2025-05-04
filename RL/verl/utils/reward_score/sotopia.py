import copy
import json
import re
from verl.envs.utils.generate import api_generate
import math
reward_prompt = """{history}

Based on previous interactions, evaluate how well participants achieve their goals. 

[Information]
Agent1: {agent1_name}
Agent1's Goal: {agent1_goal}

Agent2: {agent2_name}
Agent2's Goal: {agent2_goal}

[Requirements]
1. Please first reiterate agent's social goals. And then please provide a comprehensive analysis about the extent to which the agent has managed to achieve these goals. In the "reasoning" field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the "score" field. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates that the agent is making progress towards their social goals.
2. Please following the output format.

Here is the output schema:
```json
{{
    "agent1": {{
        "reasoning": "",
        "score": "", 
    }},
    "agent2": {{
        "reasoning": "",
        "score": "", 
    }}
}}
```
Please provide your response directly below this prompt."""


def reward_format(data):
    match = re.search(r'\{.*\}', data, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            try:
                data = json.loads(match.group(0))
            except:
                try:
                    agent1_str = re.search(r'"agent1": "(.*?)"', json_str).group(1)
                    reason1_str = re.search(r'"reasoning": "(.*?)"', agent1_str).group(1)
                    score1_str = re.search(r'"score": "(.*?)"', agent1_str).group(1)
                    agent2_str = re.search(r'"agent2": "(.*?)"', json_str).group(1)
                    reason2_str = re.search(r'"reasoning": "(.*?)"', agent2_str).group(1)
                    score2_str = re.search(r'"score": "(.*?)"', agent2_str).group(1)
                    data = {"agent1": {"reasoning": reason1_str, "score": score1_str},
                            "agent2": {"reasoning": reason2_str, "score": score2_str}}
                except:
                    return []
            assert 'agent1' in data, 'agent1 should be in the data'
            assert 'agent2' in data, 'agent2 should be in the data'
            assert 'reasoning' in data['agent1'], 'judgment should be in the data'
            assert 'reasoning' in data['agent2'], 'judgment should be in the data'
            assert 'score' in data['agent1'], 'score should be in the data'
            assert 'score' in data['agent2'], 'score should be in the data'
            data['agent1']['score'] = float(data['agent1']['score'])
            data['agent2']['score'] = float(data['agent2']['score'])
            assert 0 <= data['agent1']['score'] <= 10, 'score should be in the range of 0 to 10'
            assert 0 <= data['agent2']['score'] <= 10, 'score should be in the range of 0 to 10'
        except Exception as e:
            return []
    else:
        return []
    return data
    
def extract_rewards(data, turn):
    if turn % 2 == 0:
        return {
            'actor': float(data['agent1']['score']),
            'oppo': float(data['agent2']['score'])
        }
    else:
        return {
            'actor': float(data['agent2']['score']),
            'oppo': float(data['agent1']['score'])
        }

def history_format(data):
    result = data['result']
    try:
        response = result.split('<|begin_of_answer|>')[-1].split('<|end_of_answer|>')[0].strip()
    except Exception as e:
        print("="*20)
        print("Processing response")
        print(e)
        print(result)
        response = result.strip().split('\n')[0]
    data['response'] = response
    temp_list = copy.deepcopy(data['state_infor']['dialog'])
    temp_list[-1] += " " + response + "\n"
    history = "\n".join(temp_list).strip()
    del temp_list
    return history
    
def get_reward(datas, reward_models):
    #process batch prompts
    prompts = []
    fail_num = 0
    for item in datas:
        # skip the format error case
        if not format_check(item):
            item['reward'] = -100
            continue
        history = history_format(item)
        state_infor = item['state_infor']
        prompt = reward_prompt.format(
                history=history,
                agent1_name=state_infor['agent1_name'],
                agent1_goal=state_infor['agent1_goal'], 
                agent2_name=state_infor['agent2_name'],
                agent2_goal=state_infor['agent2_goal']
        )

        prompts.append({"prompt": {'user': prompt}})

    reward_temperature = 0.0
    configs = []
    for i, reward_model in enumerate(reward_models):
        reward_model['run_args']['temperature'] = reward_temperature
        configs.append(reward_model)

    existing_idx = [idx for idx, item in enumerate(datas) if 'reward' not in item]

    while prompts:
        if fail_num >= 2:
            for config in configs:
                config['run_args']['temperature'] = 1.0
        if fail_num == 5:
            for idx, item in enumerate(existing_idx):
                datas[item]['reward'] = None
                print(" [Error] Reward Failed Prompt: ")
                print(prompts[idx])
            return datas
        results = api_generate(prompts, configs, 30*len(configs), 8000)
        
        fail_list = []
        for i, result in enumerate(results):
            formatted_result = reward_format(result)
            if formatted_result:
                try:
                    reward = extract_rewards(formatted_result, datas[existing_idx[i]]['extra_info']['turn'])
                    datas[existing_idx[i]]['reward'] = reward
                except Exception as e:
                    fail_list.append(i)
            else:
                fail_list.append(i)
        prompts = [item for idx, item in enumerate(prompts) if idx in fail_list]
        existing_idx = [item for idx, item in enumerate(existing_idx) if idx in fail_list]
        fail_num += 1
        
    return datas

def scale_gradient(grad, current_state):
    if grad == 0:
        return 0
    upper = 10 - current_state
    lower = current_state
    return grad / upper if grad > 0 else grad / lower

def get_answer_length_score(num_tokens: int, used_tokens: int):
    alpha = 1/75
    beta = alpha

    delta = used_tokens - abs(num_tokens)
    sc = 0
    if delta < 0:
        sc = beta * delta * -1
    else:
        sc = alpha * delta * -1
    # Clip sc to [-1, 1]
    sc = max(-1, min(1, sc))
    return (sc + 1)/2

def compute_score(datas, reward_models, tokenizer):
    reward_datas = get_reward(datas, reward_models)
    scores = []
    for data in reward_datas:
        if data['reward'] == -100:
            scores.append({ "score": -2,
                            "level": -1,
                            "used_token": -1})
            continue
        if data['reward'] == None:
            scores.append({ "score": 0,                            
                            "level": -1,
                            "used_token": -1})
            continue
        turn = data['extra_info']['turn']
        rollout = data['extra_info']['rollout']
        used_token = data['used_token']
        state_reward = extract_rewards(data['state_infor']['state_reward'], turn)
        reward_dict = {
                    "actor_state": state_reward['actor'],
                    "oppo_state": state_reward['oppo'],
                    "actor_reward": data['reward']['actor'],
                    "oppo_reward": data['reward']['oppo'],
                    "actor_grad": data['reward']['actor'] - state_reward['actor'], 
                    "oppo_grad": data['reward']['oppo'] - state_reward['oppo'],
                    "turn": turn,
                    "rollout": rollout,
                    "level": data['level'],
                    "used_token": used_token,
                    "answer_token": len(tokenizer.tokenize(data['answer']))
                }
        # scaling the grad
        actor_grad = scale_gradient(reward_dict['actor_grad'], reward_dict['actor_state'])
        
        # scale to [0, 1]
        reward_dict['actor_grad_scaled'] = (actor_grad + 1) / 2
        
        sc = reward_dict['actor_grad_scaled']      
        
        reward_dict['grad_score'] = sc
        reward_dict['answer_length_score'] = get_answer_length_score(250, reward_dict['answer_token'])
        reward_dict['score'] = reward_dict['grad_score'] * reward_dict['answer_length_score']
        scores.append(reward_dict)
    return scores

def format_check(data):
    LEVEL_CONFIGS = {
        1: {
            'identifier': "Thinking Level: 1\n<|begin_of_answer|>\n",
            'tags': {
                'answer_start': ("Thinking Level: 1\n<|begin_of_answer|>\n", 1),
                'answer_end': ("\n<|end_of_answer|>", 1)
            },
            'actions': {}
        },
        2: {
            'identifier': "Thinking Level: 2\n<|begin_of_thinking|>\n",
            'tags': {
                'think_start': ("Thinking Level: 2\n<|begin_of_thinking|>\n", 1),
                'think_end': ("\n<|end_of_thinking|>", 1),
                'answer_start': ("<|begin_of_answer|>\n", 1),
                'answer_end': ("\n<|end_of_answer|>", 1)
            },
            'actions': {
                "intent_start": ("Intent:\n", 1),
                "intent_end": ("\n\nStyle:\n", 1),
                "style_start": ("Style:\n", 1),
                "style_end": ("\n\nResponse:\n", 1),
                "response_start": ("Response:\n", 1),
            }
        },
        3: {
            'identifier': "Thinking Level: 3\n<|begin_of_thinking|>\n",
            'tags': {
                'think_start': ("Thinking Level: 3\n<|begin_of_thinking|>\n", 1),
                'think_end': ("\n<|end_of_thinking|>", 1),
                'answer_start': ("<|begin_of_answer|>\n", 1),
                'answer_end': ("\n<|end_of_answer|>", 1)
            },
            'actions': {
                "history_start": ("History:\n", 1),
                "history_end": ("\n\nGoal:\n", 1),
                "goal_start": ("Goal:\n", 1),
                "goal_end": ("\n\nIntent:\n", 1),
                "intent_start": ("Intent:\n", 1),
                "intent_end": ("\n\nAssess:\n", 1),
                "assess_start": ("Assess:\n", 1),
                "assess_end": ("\n\nStrategy:\n", 1),
                "strategy_start": ("Strategy:\n", 1),
                "strategy_end": ("\n\nStyle:\n", 1),
                "style_start": ("Style:\n", 1),
                "style_end": ("\n\nResponse:\n", 1),
                "response_start": ("Response:\n", 1),
            }
        },
        4: {
            'identifier': "Thinking Level: 4\n<|begin_of_thinking|>\n",
            'tags': {
                'think_start': ("Thinking Level: 4\n<|begin_of_thinking|>\n", 1),
                'think_end': ("\n<|end_of_thinking|>", 1),
                'answer_start': ("<|begin_of_answer|>\n", 1),
                'answer_end': ("\n<|end_of_answer|>", 1)
            },
            'actions': {
                "history_start": ("History:\n", 1),
                "history_end": ("\n\nGoal:\n", 1),
                "goal_start": ("Goal:\n", 1),
                "goal_end": ("\n\nIntent:\n", 1),
                "intent_start": ("Intent:\n", 1),
                "intent_end": ("\n\nAssess:\n", 1),
                "assess_start": ("Assess:\n", 1),
                "assess_end": ("\n\nStrategy:\n", 1),
                "strategy_start": ("Strategy:\n", 1),
                "strategy_end": ("\n\nDeduction:\n", 1),
                "deduction_start": ("Deduction:\n", 1),
                "deduction_end": ("\n\nIntegration:\n", 1),
                "integration_start": ("Integration:\n", 1),
                "integration_end": ("\n\nStyle:\n", 1),
                "style_start": ("Style:\n", 1),
                "style_end": ("\n\nResponse:\n", 1),
                "response_start": ("Response:\n", 1),
            }
        },
    }
    # indentify level
    
    result = data['result']
    level = None
    config = None
    for lvl, cfg in LEVEL_CONFIGS.items():
        if cfg['identifier'] in result:
            level = lvl
            config = cfg
            break
    
    if not config:
        return False
    
    data['level'] = str(level)

    def verify_tags(tags, text):
        positions = {}
        for tag_name, (tag_str, expected_count) in tags.items():
            count = text.count(tag_str)
            positions[tag_name] = pos = text.find(tag_str)
            
            # print(f"  {tag_str}: count={count}, position={pos}")
            
            if count != expected_count:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
                return None
        return positions

        
    # verify tag and pos
    tag_positions = verify_tags(config['tags'], result)
    if not tag_positions:
        return False

    # verify tag order
    def verify_tag_order(positions, level):
        if level == 1:
            return positions['answer_start'] < positions['answer_end']
        else:
            return (positions['think_start'] < positions['think_end'] and
                   positions['think_end'] < positions['answer_start'] and
                   positions['answer_start'] < positions['answer_end'])

    if not verify_tag_order(tag_positions, level):
        print("  [Error] Incorrect tag order")
        return False
    
    def verify_answer(text):
        answer = text.split('<|begin_of_answer|>\n')[-1].split('\n<|end_of_answer|>')[0]
        if answer.count('"') != 2:
            print("  [Error] Incorrect answer format")
            return False
        if answer[0] != '"' or answer[-1] != '"':
            print("  [Error] Incorrect answer format")
            return False
        return True
    
    if not verify_answer(result):
        return False
        
    data['answer'] = result.split('<|begin_of_answer|>')[-1].split('<|end_of_answer|>')[0].strip()

    # verify actions
    if config['actions']:
        thinking = result[tag_positions['think_start']:tag_positions['think_end']]
        action_positions = verify_tags(config['actions'], thinking)
        if not action_positions:
            return False

        # verify action order
        def verify_action_order(positions, actions):
            action_names = list(actions.keys())
            for i in range(len(action_names) - 1):
                if positions[action_names[i]] > positions[action_names[i + 1]]:
                    return False
            return True

        if not verify_action_order(action_positions, config['actions']):
            print("  [Error] Incorrect action order")
            return False
    
    return data

    
    
    
    
            
        
        
        
    
    
    


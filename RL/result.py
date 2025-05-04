# -*- coding:utf-8 -*-
import sys

from collections import defaultdict
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from verl.envs.utils.utils import read_json_file


def sotopia(args):
    datas = read_json_file(args.data_path)
    hard_ids = read_json_file(args.hard_ids)
    actor_reward = {"overall": []}
    oppo_reward = {"overall": []}
    reward_avg = {"overall": []}
    
    hard_actor_reward = {"overall": []}
    hard_oppo_reward = {"overall": []}
    hard_reward_avg = {"overall": []}
    nums = 0
    hard_nums = 0

    for i, data in enumerate(tqdm(datas)):
        idx = data['env_id']
        if data['rewards'][-1] == -100 or data['agent1_scores'] == {}:
            continue
        nums += 1
        actor = data['actor_role']
        actor_score = data['agent1_scores'] if actor == 'agent1' else data['agent2_scores']
        oppo_score = data['agent1_scores'] if actor == 'agent2' else data['agent2_scores']
        actor_reward_list = []
        oppo_reward_list = []
        for key in actor_score:
            if key not in actor_reward:
                actor_reward[key] = []
            if key not in oppo_reward:
                oppo_reward[key] = []
            if key not in reward_avg:
                reward_avg[key] = []
            actor_reward[key].append(actor_score[key][-1])
            oppo_reward[key].append(oppo_score[key][-1])
            reward_avg[key].append((actor_score[key][-1] + oppo_score[key][-1])/2.0)
            if idx in hard_ids or ('hard' in data and data['hard']):
                if key not in hard_actor_reward:
                    hard_actor_reward[key] = []
                if key not in hard_oppo_reward:
                    hard_oppo_reward[key] = []
                if key not in hard_reward_avg:
                    hard_reward_avg[key] = []
                hard_actor_reward[key].append(actor_score[key][-1])
                hard_oppo_reward[key].append(oppo_score[key][-1])
                hard_reward_avg[key].append((actor_score[key][-1] + oppo_score[key][-1])/2.0)
            
            actor_reward_list.append(actor_score[key][-1])
            oppo_reward_list.append(oppo_score[key][-1])
        actor_reward["overall"].append(sum(actor_reward_list)/len(actor_reward_list))
        oppo_reward["overall"].append(sum(oppo_reward_list)/len(oppo_reward_list))
        reward_avg["overall"].append((sum(actor_reward_list) + sum(oppo_reward_list))/(len(actor_reward_list)+len(oppo_reward_list)))
        if idx in hard_ids or ('hard' in data and data['hard']):
            hard_nums += 1
            hard_actor_reward["overall"].append(sum(actor_reward_list)/len(actor_reward_list))
            hard_oppo_reward["overall"].append(sum(oppo_reward_list)/len(oppo_reward_list))
            hard_reward_avg["overall"].append((sum(actor_reward_list) + sum(oppo_reward_list))/(len(actor_reward_list)+len(oppo_reward_list)))

    print(f"Experiment Settings: {os.path.basename(args.data_path)}")
    print(f"Sotopia Total Episodes: {nums}")
    print("="*50)
    print("actor_reward:")
    for key in actor_reward:
        print(f"{key}: Mean reward: {np.mean(actor_reward[key])} | Std reward: {np.std(actor_reward[key])}")
    print("="*50)
    print("oppo_reward:")
    for key in oppo_reward:
        print(f"{key}: Mean reward: {np.mean(oppo_reward[key])} | Std reward: {np.std(oppo_reward[key])}")
    print("="*50)
    print("reward_avg:")
    for key in reward_avg:
        print(f"{key}: Mean reward: {np.mean(reward_avg[key])} | Std reward: {np.std(reward_avg[key])}")
        
    print("="*50)
    print(f"Sotopia Hard Total Episodes: {hard_nums}")
    print("="*50)
    print("hard_actor_reward:")
    for key in hard_actor_reward:
        print(f"{key}: Mean reward: {np.mean(hard_actor_reward[key])} | Std reward: {np.std(hard_actor_reward[key])}")
    print("="*50)
    print("hard_oppo_reward:")  
    for key in hard_oppo_reward:
        print(f"{key}: Mean reward: {np.mean(hard_oppo_reward[key])} | Std reward: {np.std(hard_oppo_reward[key])}")
    print("="*50)
    print("hard_reward_avg:")
    for key in hard_reward_avg:
        print(f"{key}: Mean reward: {np.mean(hard_reward_avg[key])} | Std reward: {np.std(hard_reward_avg[key])}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for result.py')
    parser.add_argument('--env', type=str, default='sotopia')
    parser.add_argument("--data_path", type=str, default="./result/sotopia.json")
    parser.add_argument("--hard_ids", type=str, default="./verl/envs/sotopia_utils/sotopia_data/sotopia_hard.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    sotopia(args)
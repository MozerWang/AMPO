"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import json
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.prompt import auto_thinking_prompt
from verl.envs.batch_sot_env import BatchSotopiaEnv

@hydra.main(config_path='config', config_name='exp_infer', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    ## verl config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    ## sotopia exp prepare
    dialog_directory = f'./result/{config.exp.env}/{config.exp.dir_name}'
    if not os.path.exists(dialog_directory):
        os.makedirs(dialog_directory)
    if config.exp.test_thinking and not config.exp.test_baseline:
        system_prompt = auto_thinking_prompt
    else: system_prompt = ""
    
    batch_size = config.data.sotopia.batch_size
    if config.exp.eval_path and config.exp.api:
        file_path = config.exp.eval_path
    else:
        file_path = os.path.join(dialog_directory, f"batch={batch_size}_temp={config.rollout.temperature}_{config.exp.exp_name}.json")
    restore = False
    existing_datas = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as fr:
            existing_datas = json.load(fr)
        if existing_datas: restore = True
    
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not config.exp.api:
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()
    else:
        wg = config.exp.model_name
        
    total_episodes = 450
    env = BatchSotopiaEnv(
        model_name= config.exp.model_name if config.exp.api else wg,
        tokenizer=tokenizer,
        env_model=config.exp.env_model,
        opponent_model=config.exp.opponent_model,
        test_same_model=config.exp.test_same_model,
        temperature=config.rollout.temperature,
        batch_size=batch_size,
        saving_path=file_path,
        system_prompt=system_prompt,
        max_turns=config.exp.max_turns,
        model_process_num=config.exp.model_process_num,
        eval_process_num=config.exp.eval_process_num,
        api=config.exp.api,
        port=config.exp.port,
        test_thinking=config.exp.test_thinking,
        test_baseline=config.exp.test_baseline,
        save_folder=dialog_directory,
        test_hard_only=config.exp.test_hard_only,
        total_episodes=total_episodes,
        rollout_prompt_length=config.rollout.prompt_length
    )
    reward_container = []
    if config.exp.test_same_model:
        progress_bar = tqdm(range(0, total_episodes, batch_size), desc="Evaluating")
    else:
        progress_bar = tqdm(range(0, total_episodes*2, batch_size), desc="Evaluating")
    for env_id_start in progress_bar:
        if env_id_start < total_episodes:
            env_ids = list(range(env_id_start, min(env_id_start + batch_size, total_episodes)))
            actor_role = 1
        else:
            env_ids = list(range(env_id_start - total_episodes, min(env_id_start + batch_size - total_episodes, total_episodes)))
            actor_role = 2
        if len(existing_datas) <= env_id_start:
            restore = False
        
        if restore:
            finish = True      
            for idx in range(len(env_ids)):
                if config.exp.test_hard_only and not existing_datas[env_id_start + idx]["hard"]:
                    continue
                if not existing_datas[env_id_start + idx]["agent1_scores"]:
                    finish = False
                    break
        else:
            finish = False
        if not finish:
            print(f"evaluating dialog from env_id {env_ids[0]} to env_id {env_ids[-1]} and actor_role {actor_role}...")
            env.reset(env_ids=env_ids, actor_role=f"agent{actor_role}", restore=restore)
            done = False
            while not done:
                rewards, done = env.batch_step()
                env.save_conversation_history()
            reward_container.extend(rewards)
            progress_bar.set_postfix_str(f"Mean reward: {np.mean(reward_container):.2f} | Std reward: {np.std(reward_container):.2f}")
        else:
            print(f"this batch has finished")
    
    env.get_final_result()

if __name__ == '__main__':
    main()
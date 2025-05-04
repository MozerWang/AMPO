import json
from tqdm import tqdm
import multiprocessing
import requests
import numpy as np
from functools import partial
from decimal import Decimal
from openai import OpenAI
import numpy as np
import sys
import time
import random

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            try:
                return str(obj, encoding='utf-8')
            except:
                return str(obj, encoding='gbk')
        elif isinstance(obj, Decimal):
            return float(obj)
        # print(obj, type(obj))
        return json.JSONEncoder.default(self, obj)


def get_api_results(prompt_input, config, port):
    prompt = prompt_input['prompt']
    if not isinstance(config, list):
        config = [config]
    config = random.sample(config, 1)[0]
    model = config['args']['api_name']
    messages = [{"role": "system", "content": prompt['system']}] if prompt.get('system') else []
    messages.append({"role": "user", "content": prompt['user']})
    
    if config['type'] == 'Azure':
        # Azure API
        headers = {"Content-Type": "application/json",
                "Authorization": config['args']['api_key']}
        raw_info = {
            "model": model,
            "messages": messages,
            "n": 1}
        raw_info.update(config['run_args'])
        try:
            callback = requests.post(config['args']['api_url'], data=json.dumps(raw_info, cls=MyEncoder), headers=headers,
                                    timeout=(60, 60))
            result = callback.json()
            # print(result)
            # todo: customize the result
            try:
                result = result['data']['response']['choices'][0]['message']['content']
            except:
                result =  result['choices'][0]['message']['content']
            return result
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'Http':
        token = config['args']['api_key']
        url = config['args']['api_url']
        payload = {
            "model": config['args']['api_name'],
            "messages": messages
        }
        payload.update(config['run_args'])
        headers = {
            'Authorization': f'{token}',
            'Content-Type': 'application/json',
        }
        try:
            resp = requests.request("POST", url, headers=headers, json=payload,timeout=(60, 60))
            result = resp.json()
            result = result['choices'][0]['message']['content']
            return result
        except Exception as e:
            return []

    
    elif config['type'] == 'OpenAI':
        # Set OpenAI's API key and API base to use vLLM's API server.
        try:
            client = OpenAI(
                api_key=config['args']['api_key'],
                base_url=config['args']['api_url'],
            )
        except:
            client = OpenAI(
                api_key=config['args']['api_key'],
                base_url=config['args']['api_url'].format(port=port),
            )
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config['run_args']['temperature'],
                timeout=60
            )
            # return chat_response.choices[0].message.model_dump()['content']
            return chat_response.choices[0].message.content 
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'gemini':
        headers = {'Content-Type': 'application/json',
                'Authorization': config['args']['api_key']}
        
        generationConfig = {}
        generationConfig.update(config['run_args'])
        raw_info = {
            "model": config['args']['api_name'],
            "contents": [{
                "parts": [
                    {
                        "text": prompt['user']
                    }
                ]}],
            "generationConfig": generationConfig
        }
        try:
            callback = requests.post(config['args']['api_url'], headers=headers, data=json.dumps(raw_info,cls=MyEncoder),
                                     timeout=(60, 60))
            result = callback.json()['candidates'][0]['content']['parts'][0]['text']
            return result
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'dashscope':
        import dashscope
        from dashscope import Generation
        from http import HTTPStatus
        model_name = config['args']['api_name']
        dashscope.api_key = config['args']['api_key']
        dashscope.base_http_api_url = config['args']['api_url']

        try:
            response = Generation.call(
                model_name,
                messages=messages,
                temperature=config['run_args']['temperature'],
                result_format='message',  # set the result to be "message"  format.
                timeout=60
            )
            return response.output.choices[0]['message']['content']
        except Exception as e:
            return []
    
    elif config['type'] == 'deepseek':
        token = config['args']['api_key']
        url = config['args']['api_url']
        payload = {
            "model": config['args']['api_name'],
            "messages": messages
        }
        payload.update(config['run_args'])
        headers = {
            'Authorization': f'{token}',
            'Content-Type': 'application/json',
        }
        try:
            resp = requests.request("POST", url, headers=headers, json=payload,timeout=(120, 120))
            result = resp.json()
            thinking_result = result['choices'][0]['message']['reasoning_content']
            result = result['choices'][0]['message']['content']
            return (result, thinking_result)
        except Exception as e:
            return []
    
    elif config['type'] == 'claude':
        headers = {"Content-Type": "application/json",
                "Authorization": config['args']['api_key']}
        raw_info = {
            "model": model,
            "messages": messages}
        raw_info.update(config['run_args'])
        try:
            callback = requests.post(config['args']['api_url'], data=json.dumps(raw_info, cls=MyEncoder), headers=headers,
                                    timeout=(60, 60))
            result = callback.json()
            # print(result)
            return result['content'][0]['text']
        except Exception as e:
            print(e)
            return []

def fetch_api_result(prompt_input, config, port, max_retries=5):
    """Attempt to get a valid result from the API, with a maximum number of retries."""
    for _ in range(max_retries):
        result = get_api_results(prompt_input, config, port)
        if result: 
            return result
        time.sleep(3)
    return None

def api(index_prompt_pair, config, port):
    index, prompt = index_prompt_pair
    response_content = fetch_api_result(prompt, config, port)
    result = response_content or ""
    return index, result

def api_generate(prompts, config, process_num, port):
    indexed_prompts = list(enumerate(prompts)) 
    func = partial(api, config=config, port=port)
    
    with multiprocessing.Pool(processes=process_num) as pool:
        results = list(tqdm(pool.imap(func, indexed_prompts), total=len(prompts), file=sys.stdout))

    results.sort(key=lambda x: x[0])
    sorted_results = [result for _, result in results]
    return sorted_results
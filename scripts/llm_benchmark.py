
import time
import numpy as np
import pandas as pd
import tqdm
import tiktoken

queries = [
    'What is the side effects of advil?',
    'Write me a Python code to calculate Fibonacci numbers.',
    # 'Are you a bot?',
    # 'Hello',
    ]

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def report_results(res):
    res['speed'] = res['total_tokens'] / (res['time'] - args.latency)
    print(res)
    print('Average speed (tokens/sec):', res['speed'].mean())

    # # (1 prompt_tokens completion_tokens) dot (latency prompt_speed completion_speed)^T = time
    # # import pdb; pdb.set_trace()
    # A = res[['prompt_tokens', 'completion_tokens']]
    # b = res['time']
    # AA = np.hstack((np.ones((A.shape[0],1)), A.to_numpy()))
    # x = np.linalg.lstsq(AA, b.to_numpy(), rcond=None)[0]
    # latency, prompt_time, completion_time = x
    # print(dict(latency=latency, prompt_speed=1.0/prompt_time, completion_speed=1.0/completion_time))

def benchmark_openai(queries):
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if args.url:
        openai.api_base = args.url

    res = []
    for query in tqdm.tqdm(queries):
        __TIC = time.time()
        resp = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                    {"role": "system", "content": ""}, # NOTE: llama-2 speed is sensitive to system prompt
                    {"role": "user", "content": query},
                ]
            )
        __TOC = time.time()
        # import pdb; pdb.set_trace()
        if args.verbose:
            print(f'User: {query}')
            print(f'Assistant: {resp.choices[0].message.content.lstrip()}')
        _res = dict(resp.usage)
        _res['model'] = resp.model
        _res['time'] = __TOC - __TIC
        res.append(_res)
    return pd.DataFrame(res)

def bendmark_tgi_llama2(queries):
    kwargs = {'max_new_tokens': 1024} # , 'return_full_text': True}
    from text_generation import Client
    client = Client(args.url, timeout=60)
    res = []
    for query in tqdm.tqdm(queries):
        __TIC = time.time()
        prompt = f"[INST] <<SYS>> <</SYS>> {query} [/INST]"
        resp = client.generate(prompt, **kwargs)
        __TOC = time.time()
        # import pdb; pdb.set_trace()
        if args.verbose:
            print(f'User: {query}')
            print(f'Assistant: {resp.generated_text.lstrip()}')
        _res = dict(prompt_tokens=num_tokens_from_string(prompt),
                   completion_tokens=num_tokens_from_string(resp.generated_text))
        _res['completion_tokens.ori'] =  resp.details.generated_tokens
        _res['total_tokens'] =  _res['prompt_tokens'] + _res['completion_tokens']
        _res['model'] = args.model
        _res['time'] = __TOC - __TIC
        res.append(_res)
    return pd.DataFrame(res)

def bendmark_tgi_phi(queries):
    kwargs = {'max_new_tokens': 1024} # , 'return_full_text': True}
    from text_generation import Client
    client = Client(args.url, timeout=60)
    res = []
    for query in tqdm.tqdm(queries):
        __TIC = time.time()
        prompt = f"{query}\n"
        resp = client.generate(prompt, **kwargs)
        __TOC = time.time()
        # import pdb; pdb.set_trace()
        if args.verbose:
            print(f'User: {query}')
            print(f'Assistant: {resp.generated_text.lstrip()}')
        _res = dict(prompt_tokens=num_tokens_from_string(prompt),
                   completion_tokens=num_tokens_from_string(resp.generated_text))
        _res['completion_tokens.ori'] =  resp.details.generated_tokens
        _res['total_tokens'] =  _res['prompt_tokens'] + _res['completion_tokens']
        _res['model'] = args.model
        _res['time'] = __TOC - __TIC
        res.append(_res)
    return pd.DataFrame(res)

def benchmark_vllm(queries):
    """Same as OpenAI, but use Completion instead of ChatCompletion. Also count w.r.t tiktoken"""
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if args.url:
        openai.api_base = args.url

    res = []
    for query in tqdm.tqdm(queries):
        __TIC = time.time()
        prompt = f"[INST] <<SYS>> <</SYS>> {query} [/INST]"
        resp = openai.Completion.create(
        # resp = openai.ChatCompletion.create(
            model=args.model,
            prompt = prompt,
            max_tokens=1024,
            )
        __TOC = time.time()
        # import pdb; pdb.set_trace()
        if args.verbose:
            print(f'User: {query}')
            print(f'Assistant: {resp.choices[0].text.lstrip()}')
        _res = dict(prompt_tokens=num_tokens_from_string(prompt),
                   completion_tokens=num_tokens_from_string(resp.choices[0].text.lstrip()))
        # _res = dict(resp.usage)
        _res['total_tokens'] =  _res['prompt_tokens'] + _res['completion_tokens']
        _res['prompt_tokens.ori'] = resp.usage['prompt_tokens']
        _res['completion_tokens.ori'] = resp.usage['completion_tokens']
        _res['model'] = resp.model
        _res['time'] = __TOC - __TIC
        res.append(_res)
    return pd.DataFrame(res)
        
def parse_args():
    import argparse
    class UltimateHelpFormatter(
        argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass
    usage = """Examples:

# For OpenAI
python llm_benchmark.py openai gpt-3.5-turbo
python llm_benchmark.py openai gpt-4

# For vLLM
python llm_benchmark.py vllm meta-llama/Llama-2-7b-chat-hf --url http://172.31.91.92:8071/v1
python llm_benchmark.py vllm meta-llama/Llama-2-70b-chat-hf --url http://172.31.91.92:8073/v1

# For HuggingFace TGI
python llm_benchmark.py tgi llama-2-7b-chat --url http://172.31.91.91:8091
# python llm_benchmark.py tgi llama-2-13b-chat --url http://172.31.91.91:8092
python llm_benchmark.py tgi llama-2-70b-chat --url http://172.31.91.91:8093
"""
    parser = argparse.ArgumentParser(
        description='LLM benchmark',
        epilog=usage,
        formatter_class=UltimateHelpFormatter, # argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('engine', choices=['openai', 'tgi', 'tgi_phi', 'vllm'],
            help='LLM engine: e.g. openai, tgi, etc.')
    parser.add_argument('model',
            help='LLM model: e.g. gpt-3.5-turbo, etc.')
    parser.add_argument('--url',
            help='LLM model: e.g. gpt-3.5-turbo, etc.')
    parser.add_argument('--latency', default=0.0, type=float,
            help='More accurate calculation is latency is known.')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Verbose.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.engine == 'openai':
        res = benchmark_openai(queries)

    elif args.engine == 'tgi':
        res  = bendmark_tgi_llama2(queries)

    elif args.engine == 'tgi_phi':
        res = bendmark_tgi_phi(queries)

    elif args.engine == 'vllm':
        res = benchmark_vllm(queries)

    report_results(res)
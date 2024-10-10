#!/usr/bin/env python3

import sys
import readline
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from typing import List, Dict
from argparse import ArgumentParser

from transformers.utils import logging
logging.set_verbosity_error() 


def main():
    argparser = ArgumentParser("a simple test script for LLM-jp")
    argparser.add_argument("-d", "--debug", action='store_true')
    argparser.add_argument("-m", "--model", choices=['llm-jp-3-1.8b', 'llm-jp-3-1.8b-instruct', 'llm-jp-3-3.7b', 'llm-jp-3-3.7b-instruct', 'llm-jp-3-13b', 'llm-jp-3-13b-instruct'], required=True, help='model variants')
    argparser.add_argument("--no-instruct", action='store_true')
    argparser.add_argument("--no-system-prompt", action='store_true')
    argparser.add_argument("--max-tokens", type=int, default=512)
    args = argparser.parse_args()

    def run_interactive(modelname):
        tokenizer = AutoTokenizer.from_pretrained("llm-jp/" + modelname)
        model = AutoModelForCausalLM.from_pretrained("llm-jp/" + modelname, device_map="auto", torch_dtype="auto")
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
    
        generation_params = {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_new_tokens": args.max_tokens,
            "repetition_penalty": 1.1
        }
    
        tokenizer.chat_template = "{{'以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n'}}{% for message in messages %}{% if message['role'] == 'user' %}{{'### 指示:\n' + message['content'] + '\n\n'}}{% endif %}{% if message['role'] == 'assistant' %}{{'### 応答:\n' + message['content'] + '\n\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### 応答:\n' }}{% endif %}"
    
        def query (
                user_query: str,
                history: List[Dict[str, str]]=None,
            ):
        
            start = time.process_time()
            # messages
            messages = ""
            if not args.no_instruct:
                messages = []
                if not args.no_system_prompt:
                    messages = [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    ]
                user_messages = [
                    {"role": "user", "content": user_query}
                ]
            else:
                user_messages = user_query
            if history:
                user_messages = history + user_messages
            messages += user_messages
        
            # generation prompts
            if not args.no_instruct:
                prompt = tokenizer.apply_chat_template(
                    conversation=messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt = messages
        
            input_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )

            if args.debug:
                print("--- prompt")
                print(prompt)
                print("--- output")

            # 推論
            output_ids = model.generate(
                input_ids.to(model.device),
                streamer=streamer,
                **generation_params,
				pad_token_id=tokenizer.eos_token_id
            )
            output = tokenizer.decode(
                output_ids[0][input_ids.size(1) :],
                skip_special_tokens=True
            )
            if not args.no_instruct:
                user_messages.append(
                    {"role": "assistant", "content": output}
                )
            else:
                user_messages += output
            end = time.process_time()
            ##
            input_tokens = len(input_ids[0])
            output_tokens = len(output_ids[0][input_ids.size(1) :])
            total_time = end - start
            tps = output_tokens / total_time

            if args.debug:
                print(f"prompt tokens = {input_tokens:.7g}")
                print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
                print(f"   total time = {total_time:f} [s]")

            return user_messages


        history = None
        while True:
            text = ""
            while True:
                value = input("input> ")
                if value:
                    text += value + "\n"
                else:
                    break
    
            if text == "":
                break
            else:
                history = query(text.rstrip("\n"), history)
    
        return 0



    run_interactive (args.model)

    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        raise e

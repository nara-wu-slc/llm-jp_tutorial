#!/usr/bin/env python3

import codecs
import re
import readline
import sys
import time
from argparse import ArgumentParser
from typing import Dict, List

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

downloaded_models = [
    'llm-jp-4.1-8b-thinking-gguf',
    'llm-jp-4.1-32b-a3b-thinking-gguf',
    'llm-jp-4.1-33b-thinking-gguf',
    'llm-jp-4.1-8b-thinking',
    'llm-jp-4.1-32b-a3b-thinking',
    'llm-jp-4.1-33b-thinking',
]


def build_harmony_prompt(messages: List[Dict[str, str]], reasoning_effort: str = "medium") -> str:
    parts = []
    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        parts.append(f"<|start|>system<|message|>Reasoning: {reasoning_effort}<|end|>")
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            if "Reasoning:" not in content:
                content = f"{content}\n\nReasoning: {reasoning_effort}"
            parts.append(f"<|start|>system<|message|>{content}<|end|>")
        elif role == "user":
            parts.append(f"<|start|>user<|message|>{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|return|>")
    parts.append("<|start|>assistant")
    return "".join(parts)


def extract_final_content(raw_output: str) -> str:
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        raw_output,
         flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"<\|[^|]+\|>", "", raw_output)
    return cleaned.strip()


def main():
    argparser = ArgumentParser("a simple test script for LLM-jp GGUF models")
    argparser.add_argument("-d", "--debug", action='store_true')
    argparser.add_argument("-m", "--model", choices=downloaded_models, required=True, help='GGUF model variants')
    argparser.add_argument("-q", "--quant", default="Q4_K_M", help='quantization type (default: Q4_K_M)')
    argparser.add_argument("-ngl", "--n-gpu-layers", type=int, default=-1, help='number of layers to offload to GPU (-1 for all)')
    argparser.add_argument("-c", "--ctx-size", type=int, default=4096, help='context size (default: 4096)')
    argparser.add_argument("--no-instruct", action='store_true')
    argparser.add_argument("--no-system-prompt", action='store_true')
    argparser.add_argument("--max-tokens", type=int, default=2048)
    argparser.add_argument("--reasoning-effort", choices=['low', 'medium', 'high'], default='medium', help='reasoning effort for thinking models')
    args = argparser.parse_args()

    def run_interactive(modelname: str):
        base_name = modelname[:-5] if modelname.endswith("-gguf") else modelname
        repo_id = f"llm-jp/{base_name}-gguf"
        filename = f"{base_name}-{args.quant}.gguf"

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx_size,
            verbose=args.debug,
        )

        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
        chat_template_str = llm.metadata.get("tokenizer.chat_template")

        def apply_template(messages: List[Dict[str, str]]) -> str:
            if chat_template_str:
                try:
                    from jinja2.sandbox import ImmutableSandboxedEnvironment
                    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                    template = env.from_string(chat_template_str)
                    return template.render(
                        messages=messages,
                        add_generation_prompt=True,
                        reasoning_effort=args.reasoning_effort,
                    )
                except Exception:
                    pass
            return build_harmony_prompt(messages, reasoning_effort=args.reasoning_effort)

        def query(
            user_query: str,
            history: List[Dict[str, str]] = None,
        ):
            start = time.perf_counter()
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

            if not args.no_instruct:
                prompt = apply_template(messages)
            else:
                prompt = messages

            add_bos = not prompt.startswith("<|startoftext|>")
            input_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=add_bos, special=True)

            if args.debug:
                print("--- prompt")
                print(prompt)
                print("--- output")

            eos_token_id = llm.token_eos()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            output_ids = []
            raw_chunks = []

            for token_id in llm.generate(
                input_ids,
                top_k=40,
                top_p=0.95,
                temp=0.8,
                repeat_penalty=1.1,
            ):
                if token_id == eos_token_id:
                    break
                output_ids.append(token_id)
                token_bytes = llm.detokenize([token_id], special=True)
                text_piece = decoder.decode(token_bytes, final=False)
                if text_piece:
                    print(text_piece, end="", flush=True)
                    raw_chunks.append(text_piece)
                if len(output_ids) >= args.max_tokens:
                    break

            tail = decoder.decode(b"", final=True)
            if tail:
                print(tail, end="", flush=True)
                raw_chunks.append(tail)
            print("", flush=True)

            raw_output = "".join(raw_chunks)
            if not args.no_instruct:
                assistant_content = extract_final_content(raw_output)
                user_messages.append(
                    {"role": "assistant", "content": assistant_content}
                )
            else:
                user_messages += raw_output

            end = time.perf_counter()
            input_tokens = len(input_ids)
            output_tokens = len(output_ids)
            total_time = end - start
            tps = output_tokens / total_time if total_time > 0 else 0.0

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

    run_interactive(args.model)
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        raise e

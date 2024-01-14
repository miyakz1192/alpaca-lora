#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser(description='Swallow利用のサンプル')

parser.add_argument('--lora_data', help='LoRAデータのディレクトリ名(絶対パスで指定する)')
parser.add_argument('--instruction', help='指示プロンプトを記載したファイル')
parser.add_argument('--input', help='入力プロンプトを記載したファイル')
parser.add_argument('--model', help='モデル名を指定します')

parser.add_argument('--verbose', action='store_true', help='詳細出力を有効')
parser.add_argument(
        '--max_new_tokens',
        type=int,
        help='出力のトークン数(デフォルト128。min:128, max:1024, step:64)')


args = parser.parse_args()


# LoRAを反映するか
do_Lora = False
if args.lora_data is not None:
    do_Lora = True

model_table = [
    "tokyotech-llm/Swallow-7b-instruct-hf",
    "tokyotech-llm/Swallow-13b-instruct-hf",
    "tokyotech-llm/Swallow-70b-instruct-hf",
    "tokyotech-llm/Swallow-7b-hf",
    "tokyotech-llm/Swallow-13b-hf",
    "tokyotech-llm/Swallow-70b-hf",
    ]

if args.verbose:
    print(f"LoRAのデータ指定={do_Lora}")
    print(f"LoRAのデータディレクトリ={args.lora_data}")
    print(f"指示プロンプトファイル={args.instruction}")
    print(f"プロンプトファイル={args.input}")
    print(f"モデルの個別指定={args.model}")

if args.model is not None:
    model_name = args.model
else:
    model_name = model_table[0]

if args.verbose:
    print(f"最終的に選択したモデル={model_name}")

tokenizer_model_name = model_name

if args.verbose:
    print(f"ベースモデルのロード")

# ここで、torch_dtype=torch.float32を設定しないとCPU環境だとエラーになる
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        # torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        device_map="auto")

# LoRA分の重みを追加。以下を参照。
# https://qiita.com/iss-f/items/9ab11ed38dde2fc1f43b
if do_Lora is True:
    if args.verbose:
        print("LoRA分の重みの加算")

    # ベースモデルのロードと同様にCPU環境だと、torch_dtype=torch.float32を指定
    model = PeftModel.from_pretrained(
        model,
        args.lora_data,
        # torch_dtype=torch.float16,
        torch_dtype=torch.float32,
        load_in_8bit=False,
    )
    # ベースモデルとLoRAの重みをマージしないと上手く動作しない。
    model = model.merge_and_unload()

PROMPT_DICT = {
    "prompt_input": (
        "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"

    ),
    "prompt_no_input": (
        "以下に、あるタスクを説明する指示があります。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    ),
}


def read_file(file_path):
    content = None

    if file_path is None:
        return content

    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read()

    return content


def create_prompt(instruction, input=None):
    if input:
        # Use the 'prompt_input' template when additional input is provided
        return PROMPT_DICT["prompt_input"].format(
                instruction=instruction, input=input)
    else:
        # Use the 'prompt_no_input' template
        # when no additional input is provided
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


Do_sample = True  # @param {type:"boolean"
temperature = 0.99  # @param {type:"slider", min:0, max:2, step:0.1}
top_p = 0.95  # @param {type:"slider", min:0, max:1, step:0.01}

max_new_tokens = 128  # @param {type:"slider", min:128, max:1024, step:64}
if args.max_new_tokens is not None:
    max_new_tokens = args.max_new_tokens

instruction_example = read_file(args.instruction)
input_example = read_file(args.input)

prompt = create_prompt(instruction_example, input_example)

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

tokens = model.generate(
    input_ids=input_ids.to(device=model.device),
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=Do_sample,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)

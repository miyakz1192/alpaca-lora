#!/usr/bin/env python3

# @title Step.2 Tokenizer & Model Loading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# LoRAを反映するか
do_Lora = True 
#do_Lora = False 
# do_LoraがTrueの時に指定するモード(Qiitaの参照先の記事のどのコーディングか)
do_iss_f = True 
do_taka  = False 

#import pdb
#pdb.set_trace()

print(f"LoRA mode = {do_Lora}")

# @markdown [https://huggingface.co/tokyotech-llm](https://huggingface.co/tokyotech-llm) から利用したいモデルを選択してください。最初は 7b-instruct から始めるのがおすすめです。13bは ColabPro では動いています。70bはColabProでもダウンロードが難しいです。

#tokenizer_model_name = "tokyotech-llm/Swallow-13b-instruct-hf" # @param ['tokyotech-llm/Swallow-7b-hf','tokyotech-llm/Swallow-7b-instruct-hf','tokyotech-llm/Swallow-13b-hf','tokyotech-llm/Swallow-13b-instruct-hf','tokyotech-llm/Swallow-70b-hf','tokyotech-llm/Swallow-70b-instruct-hf']
#model_name = "tokyotech-llm/Swallow-13b-instruct-hf" # @param ['tokyotech-llm/Swallow-7b-hf','tokyotech-llm/Swallow-7b-instruct-hf','tokyotech-llm/Swallow-13b-hf','tokyotech-llm/Swallow-13b-instruct-hf','tokyotech-llm/Swallow-70b-hf','tokyotech-llm/Swallow-70b-instruct-hf']
#


# 70bにチャレンジ！(とりあえず動作)
#tokenizer_model_name = "tokyotech-llm/Swallow-70b-instruct-hf" 
#model_name = "tokyotech-llm/Swallow-70b-instruct-hf" 

# 普段使いは7b
tokenizer_model_name = "tokyotech-llm/Swallow-7b-instruct-hf" 
model_name = "tokyotech-llm/Swallow-7b-instruct-hf" 

# Load
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
        torch_dtype=torch.float32, 
        #torch_dtype=torch.float16, 
        low_cpu_mem_usage=True ,
        load_in_8bit=False,
        device_map="auto")

#LoRA分を追加  以下を参照。
#https://qiita.com/iss-f/items/9ab11ed38dde2fc1f43b
# import pdb
# pdb.set_trace()
#LORA_WEIGHTS = './alpaca-lora/output/checkpoint-200/'

if do_Lora == True and do_iss_f == True:
    print("INFO: Doing LoRA weight(do_iss_f)")
    LORA_WEIGHTS = './alpaca-lora/output/'
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        #torch_dtype=torch.float16,
        torch_dtype=torch.float32,
        load_in_8bit=False,
    )
    model = model.merge_and_unload()

if do_Lora == True and do_taka == True:
    print("INFO: Doing LoRA weight(do_taka) => Do nothing")
    pass

# @title
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

def create_prompt(instruction, input=None):
    """
    Generates a prompt based on the given instruction and an optional input.
    If input is provided, it uses the 'prompt_input' template from PROMPT_DICT.
    If no input is provided, it uses the 'prompt_no_input' template.

    Args:
        instruction (str): The instruction describing the task.
        input (str, optional): Additional input providing context for the task. Default is None.

    Returns:
        str: The generated prompt.
    """
    if input:
        # Use the 'prompt_input' template when additional input is provided
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        # Use the 'prompt_no_input' template when no additional input is provided
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


# @title Step.3 Settings & Prompts
instruction_example = "以下のトピックに関する詳細な情報を提供してください。" # @param {type: "string"}
#input_example = "\u6771\u4EAC\u5DE5\u696D\u5927\u5B66\u306E\u6A2A\u6D5C\u306B\u3042\u308B\u30AD\u30E3\u30F3\u30D1\u30B9\u306E\u98DF\u5802\u306B\u3064\u3044\u3066\u8AAC\u660E\u3057\u3066\u304F\u3060\u3055\u3044" # @param {type: "string"}
input_example = "東京工業大学の横浜にあるキャンパスの食堂について説明してください"
input_example = "小森田友明はいつ生まれたの？"
input_example = "アリスの両親には3人の娘がいる：エイミー、ジェシー、そして三女の名前は？"
input_example = "ステイルメイトの時に、私の方が多くの駒を持っていたら、私の勝ちですか？"
input_example = "ロラパルーザについての参考文章が与えられ、どこで行われ、誰が始めたのか、何なのか？"
input_example =  "カイル・ヴァンジルがチーム61得点のうち36得点を挙げたとき、誰と対戦していたのですか？"
input_example = "この中でラッパーなのはどれ？エミネム、マイケル・ジャクソン、リアーナ、50セント"

Do_sample=True #@param {type:"boolean"}

if Do_sample:
  temperature = 0.99 #@param {type:"slider", min:0, max:2, step:0.1}
  top_p = 0.95 #@param {type:"slider", min:0, max:1, step:0.01}

max_new_tokens=128 #@param {type:"slider", min:128, max:1024, step:64}

# Example usage
# instruction_example = "以下のトピックに関する詳細な情報を提供してください。"
# input_example = "東京工業大学の主なキャンパスについて教えてください"
prompt = create_prompt(instruction_example, input_example)

input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

# tokens = model.generate(
#     input_ids.to(device=model.device),
#     max_new_tokens=max_new_tokens,
#     temperature=temperature,
#     top_p=top_p,
#     do_sample=Do_sample,
# )


print("================================")
print("INFO: Go to model.generate")
print("================================")

tokens = model.generate(
    input_ids=input_ids.to(device=model.device),
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=Do_sample,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)

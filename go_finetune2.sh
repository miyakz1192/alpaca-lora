set -x
python3 finetune.py --base_model="tokyotech-llm/Swallow-7b-instruct-hf"  --data_path="./dataset/databricks-dolly-15k-ja.json" --num_epochs=1 --output_dir=output

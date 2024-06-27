import easyjailbreak
import torch
import logging
from transformers import set_seed

# Set up logging format and seed
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
set_seed(42)

# Load model and tokenizer
model = easyjailbreak.models.from_pretrained(
    '/root/autodl-tmp/vicuna-7b-v1.5', 
    'vicuna_v1.1', 
    dtype=torch.bfloat16, 
    max_new_tokens=40)

# Load dataset
# dataset_name = 'AdvBench'
# dataset = easyjailbreak.datasets.JailbreakDataset(dataset_name)

path = "/root/autodl-tmp/EasyJailbreak-EAI/datasets/advbench.csv"
dataset = easyjailbreak.datasets.JailbreakDataset.load_csv(path)[:5]

# API class initialization and settings
attacker = easyjailbreak.attacker.GCGEAI(
    attack_model=model,
    target_model=model,
    jailbreak_datasets=dataset,
    jailbreak_prompt_length=20,
    num_turb_sample=512,
    batchsize=256,      # decrease if OOM
    top_k=256,
    max_num_iter=200
)

# Launch the attack
attacker.attack()

output_file = 'gcg_result.jsonl'
attacker.jailbreak_datasets.save_to_jsonl(output_file)
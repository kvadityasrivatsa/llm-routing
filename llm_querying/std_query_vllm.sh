#!/bin/bash
#SBATCH --output=./slurm_ledger/slurm-%A.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=50G                   # Total RAM to be used
#SBATCH --cpus-per-task=16          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

# As you command Sire!

nvidia-smi

#MODEL_NAME="llama2-13b-chat"
#MODEL_NAME="mistral-7b-inst"
#MODEL_NAME="metamath-13b"
#MODEL_NAME="gemma-7b-it"
MODEL_NAME="qwen2-7b-chat"
#MODEL_NAME="llama2-70b-chat"

#MODEL_NAME="llama2-13b-lm"
#MODEL_NAME="mistral-7b-lm"
#MODEL_NAME="falcon-7b"
#MODEL_NAME="vicuna-13b"

TEMPERATURE="0.7"
REPETITION_PENALTY="1.0" 
MAX_LEN="750"
CHAT_MODE="1"   # keep 0
QUERY_FIELD="zero_shot_cot_prompt"
#QUERY_PATH="./consolidated_datasets/gsm8k.json"
QUERY_PATH="./consolidated_datasets/mmlu.json"
_UUID="$(uuidgen -r)"
echo "_UUID"

python3 std_query_vllm.py \
--model-name $MODEL_NAME \
--batch-size 1 \
--query-limit -1 \
--n-seq 5 \
--seed $RANDOM \
--temperature $TEMPERATURE \
--repetition-penalty $REPETITION_PENALTY \
--uuid $_UUID \
--max-len $MAX_LEN \
--chat-mode $CHAT_MODE \
--query-field $QUERY_FIELD \
--query-paths $QUERY_PATH

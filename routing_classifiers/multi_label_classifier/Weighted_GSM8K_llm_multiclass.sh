#GSM8K
export train_data="/root/llm_classifier/data/gsm8k/train_all.csv"
export validation_data="/root/llm_classifier/data/gsm8k/val_all.csv"
export test_data="/root/llm_classifier/data/gsm8k/test_all.csv"
export output_dir="/root/llm_classifier/output/gsm8k/exp_15"

#export modelcheckpoint="bert-base-uncased"
#export modelcheckpoint="roberta-base"
export modelcheckpoint="distilbert/distilbert-base-uncased"
#export modelcheckpoint="google-t5/t5-base"
#export modelcheckpoint="tbs17/MathBERT"      #https://huggingface.co/tbs17/MathBERT
#export modelcheckpoint="AnReu/math_albert"  #https://huggingface.co/AnReu/math_albert

python -u weighted_llm_multiclassifier.py \
    --model_name_or_path ${modelcheckpoint} \
    --train_file ${train_data} \
    --validation_file ${validation_data} \
    --test_file ${test_data} \
    --shuffle_train_dataset \
    --text_column_name "question" \
    --label_column_name "maj_6_model" \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-6 \
    --num_train_epochs 15 \
    --do_predict \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --output_dir ${output_dir} \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --overwrite_output_dir True \
    --evaluation_strategy steps \
    
   
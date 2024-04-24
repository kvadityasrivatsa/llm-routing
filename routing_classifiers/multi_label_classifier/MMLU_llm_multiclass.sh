#MMLU
export train_data="/root/llm_classifier/data/mmlu/train_all_top.csv"
export validation_data="/root/llm_classifier/data/mmlu/val_all_top.csv"
export test_data="/root/llm_classifier/data/mmlu/test_all_top.csv"
export output_dir="/root/llm_classifier/output/mmlu/exp_12"

#export modelcheckpoint="bert-base-uncased"
export modelcheckpoint="roberta-base"
#export modelcheckpoint="google-t5/t5-base"
#export modelcheckpoint="tbs17/MathBERT"      #https://huggingface.co/tbs17/MathBERT
#export modelcheckpoint="AnReu/math_albert"  #https://huggingface.co/AnReu/math_albert

python -u llm_multiclassifier.py \
    --model_name_or_path ${modelcheckpoint} \
    --train_file ${train_data} \
    --validation_file ${validation_data} \
    --test_file ${test_data} \
    --shuffle_train_dataset \
    --text_column_name "question" \
    --label_column_name "maj_2_model" \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --do_predict \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --overwrite_output_dir True \
    --output_dir ${output_dir} \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
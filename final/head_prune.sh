esp_dir=${HOME}/espnet
eg_dir=${esp_dir}/egs2/librispeech/asr1_head_pruning


python3 asr_prune.py \
        --token_list /dev/null \
        --train_data_path_and_name_and_type "${eg_dir}/dump/raw/train_100/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "${eg_dir}/dump/raw/train_100/text,text,text" \
        --train_shape_file "${eg_dir}/exp/asr_stats_raw_en_bpe5000/train/speech_shape" \
        --train_shape_file "${eg_dir}/exp/asr_stats_raw_en_bpe5000/train/text_shape.bpe" \
        --output_dir /dev/null \
        --pretrained_model_name "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best" \
        --fold_length 80000 --fold_length 150 \
        --token_list "${eg_dir}/data/en_token_list/bpe_unigram5000/tokens.txt" \
        --bpemodel "${eg_dir}/data/en_token_list/bpe_unigram5000/bpe.model" \
        --path_head_grad "./head_grad.pt" \
        --ngpu 1 \
        --batch_size 20

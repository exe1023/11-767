#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_100"
valid_set="dev"
test_sets="test_clean dev_clean"

#asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
#asr_config=conf/tuning/train_layer_pruning.yaml
asr_config=conf/tuning/train_layer_pruning_finetune.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

#./asr.sh \
#    --lang en \
#    --ngpu 2 \
#    --nbpe 5000 \
#    --max_wav_duration 30 \
#    --asr_config "${asr_config}" \
#    --lm_config "${lm_config}" \
#    --inference_config "${inference_config}" \
#    --train_set "${train_set}" \
#    --valid_set "${valid_set}" \
#    --test_sets "${test_sets}" \
#    --stop_stage 4

pretrained_path="/usr0/home/yitingye/miniconda3/envs/odml/lib/python3.7/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e"
bpemodel="${pretrained_path}/data/token_list/bpe_unigram5000/bpe.model"
mkdir -p data/en_token_list/bpe_unigram5000/

python get_token_list.py "${pretrained_path}/exp/asr_train_asr_transformer_e18_raw_bpe_sp/config.yaml"

cp ./token_list.txt data/en_token_list/bpe_unigram5000/tokens.txt
cp ${bpemodel} data/en_token_list/bpe_unigram5000/

# Note: need to run from stage 8 for one time to generate trian data

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --stage 11 \
    --stop_stage 11

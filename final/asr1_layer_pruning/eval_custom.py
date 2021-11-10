# Initialize models
import time
import torch
import string
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import pandas as pd
import soundfile
from tqdm import tqdm
from jiwer import wer
import numpy as np

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))


lang = 'en'
fs = 16000 #@param {type:"integer"}
#model_dir = 'exp/asr_train_layer_pruning_raw_en_bpe5000/'
#config = 'config.yaml'
#ckpt = 'valid.acc.ave_10best.pth'

model_dir = 'exp/asr_train_layer_pruning_finetune_raw_en_bpe5000/'
config = 'config.yaml'
ckpt = '2epoch.pth'

ctc_weights = [0, 0.3, 1.0]

egs = pd.read_csv("LibriSpeech/test_clean.csv")
for ctc_weight in ctc_weights:
    print(f'CTC weight: {ctc_weight}')
    
    speech2text = Speech2Text(asr_train_config = f'{model_dir}/{config}',
                              asr_model_file = f'{model_dir}/{ckpt}',
                              ctc_weight= ctc_weight,
                              device='cuda')

    num_params = sum([np.prod(p.size()) for p in speech2text.asr_model.parameters()])
    print(num_params)
    # inference on whole dataset
    refs, hyps = [], []
    for index, row in tqdm(egs.iterrows(), total=len(egs.index)):
        speech, rate = soundfile.read(row["path"])
        assert fs == int(row["sr"])
        nbests = speech2text(speech)
        if index == 100:
            break

        text, *_ = nbests[0]

        refs.append(text_normalizer(row['text']))
        hyps.append(text_normalizer(text))

    print(refs, hyps)
    print(wer(refs, hyps))

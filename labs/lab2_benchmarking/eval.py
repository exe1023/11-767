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

lang = 'en'
fs = 16000 #@param {type:"integer"}

models = [
    #'Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
    #'kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best',
    #'byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp',
    #'kamo-naoyuki/wsj',
    'kamo-naoyuki/aishell_conformer'
]
ctc_weights = [0, 0.3, 1.0]

#tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave' #@param ["Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave", "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"] {type:"string"}

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

egs = pd.read_csv("LibriSpeech/test_clean.csv")
for tag in models:
    print(f'Evaling {tag}')
    for ctc_weight in ctc_weights:
        print(f'CTC weight: {ctc_weight}')
        d = ModelDownloader()
        # It may takes a while to download and build models
        speech2text = Speech2Text(
            **d.download_and_unpack(tag),
            device="cuda",
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=ctc_weight,
            beam_size=5,
            penalty=0.0,
            batch_size=1,
            nbest=1
        )
        num_params = sum([np.prod(p.size()) for p in speech2text.asr_model.parameters()])
        print(num_params)
        # inference on whole dataset
        refs, hyps = [], []
        for index, row in tqdm(egs.iterrows(), total=len(egs.index)):
            speech, rate = soundfile.read(row["path"])
            assert fs == int(row["sr"])
            nbests = speech2text(speech)

            text, *_ = nbests[0]

            refs.append(text_normalizer(row['text']))
            hyps.append(text_normalizer(text))

        print(wer(refs, hyps))

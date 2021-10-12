import argparse
import ipdb
import logging
import os
import sys
import string
import sklearn.decomposition
import soundfile
from espnet2.bin.asr_inference import Speech2Text
import pandas as pd
from timeit import default_timer as timer
import random
from tqdm import tqdm
import json
import torch


dir_librispeech = '../raw_data/LibriSpeech'


def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))


def main(args):
    models = [
        "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
        'kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best',
    ]

    logs = {}
    for model in models:
        logs[model] = {}
        for ctc in [0, 0.3, 1.0]:
            speech2text = Speech2Text.from_pretrained(
                model,
                maxlenratio=0.0,
                minlenratio=0.0,
                beam_size=5,
                ctc_weight=ctc,
                penalty=0.0,
                nbest=1,
                batch_size=1
            )

            torch.backends.quantized.engine = 'qnnpack'
            asr_model = speech2text.asr_model
            beam = speech2text.beam_search
            quantized_model = torch.quantization.quantize_dynamic(
                asr_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            quantized_beam = torch.quantization.quantize_dynamic(
                beam, {torch.nn.Linear}, dtype=torch.qint8
            )
            speech2text.asr_model = quantized_model
            speech2text.beam_search = quantized_beam

            print(f'Loaded {model}, c = {ctc} .')

            total_time = 0
            time_lens = []
            # inference on whole dataset
            refs, hyps = [], []
            egs = pd.read_csv(f'{dir_librispeech}/test_clean.csv')
            random.seed(524)
            egs = random.sample(list(egs.iterrows()), 100)

            for index, (_, row) in enumerate(tqdm(egs)):
                speech, rate = soundfile.read(row["path"])
                # assert fs == int(row["sr"])

                start = timer()
                nbests = speech2text(speech)
                end = timer()
                total_time += (end - start)

                text, *_ = nbests[0]

                refs.append(text_normalizer(row['text']))
                hyps.append({text_normalizer(text)})

                time_lens.append(
                    (end - start, len(speech))
                )

            logs[model][ctc] = time_lens
            with open('/tmp/log.json', 'w') as f:
                json.dump(logs, f, indent='  ')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('arg1', type=None,
    #                     help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main(args)

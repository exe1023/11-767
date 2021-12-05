import torch
import argparse
import ipdb
import logging
import os
import sys
import string
import soundfile
import random
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from espnet2.bin.asr_inference import Speech2Text
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from jiwer import wer


def main(args):
    print('START', flush=True)
    random.seed(524)

    egs = pd.read_csv(f'{args.dir_librispeech}/test_clean.csv')
    rows = list(egs.iterrows())
    if args.n_sample is not None:
        rows = random.sample(rows, args.n_sample)

    logs = {}
    for trial in range(args.n_trial):
        name_or_path = None if args.path_model is not None \
            else args.pretrained_model_name
        speech2text = Speech2Text.from_pretrained(
            name_or_path,
            asr_model_file=args.path_model,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=args.beam_size,
            ctc_weight=1.0,
            penalty=0.0,
            nbest=1,
            batch_size=1,
            lm_weight=args.lm_weight,
            ngram_weight=args.ngram_weight,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model = speech2text.asr_model
        n_layer = len(model.encoder.encoders)

        # sample a set of layers until finding a set that hasn't been tested.
        while True:
            skip_layers = random.sample(list(range(n_layer)), args.n_layer)
            skip_layers = tuple(sorted(skip_layers))
            if skip_layers not in logs:
                break

        model.encoder.encoders = MultiSequential(*[
            layer
            for i, layer in enumerate(
                    model.encoder.encoders
            )
            if i not in skip_layers
        ])

        refs = []
        hyps = []
        times = []

        for index, (_, row) in enumerate(tqdm(rows)):
            speech, rate = soundfile.read(
                args.dir_librispeech / row["path"]
            )

            start = timer()
            nbests = speech2text(speech)
            end = timer()
            times.append(end - start)

            text, *_ = nbests[0]

            refs.append(text_normalizer(row['text']))
            hyps.append(text_normalizer(text))

        times = torch.tensor(times)
        er = wer(refs, hyps)
        print(
            f'skip layers: {skip_layers}',
            flush=True
        )
        print(
            f'Time: {times.mean().item():.3f} ({times.std().item():.3f})',
            flush=True
        )
        print(f'WER: {er}', flush=True)
        logs[skip_layers] = {
            'time': times.mean(),
            'wer': er
        }

        min_er, min_sl = 1, None
        for sl, log in logs.items():
            if log['wer'] < min_er:
                min_er = log['wer']
                min_sl = sl
        print(f'min set = {min_sl} ({min_er})', flush=True)

    print(json.dumps(logs, indent=' '))


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--base_dir', type=Path, default=Path('./'),
                        help='')
    parser.add_argument(
        '--pretrained_model_name', type=str,
        default='Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw'
        '_bpe_sp_valid.acc.best',
        help=''
    )
    parser.add_argument('--path_model', type=str, default=None)
    parser.add_argument(
        '--dir_librispeech', type=Path,
        default=Path('~/espnet/data/librispeech/LibriSpeech/').expanduser()
    )
    parser.add_argument(
        '--n_sample', type=int, default=100
    )
    parser.add_argument(
        '--normalize', type=str, default='l2'
    )
    parser.add_argument(
        '--beam_size', type=int, default=1
    )
    parser.add_argument(
        '--lm_weight', type=float, default=1.0
    )
    parser.add_argument(
        '--rank', type=str, default='importance'
    )
    parser.add_argument(
        '--truncate_layer', type=int, default=None
    )
    parser.add_argument(
        '--skip_layers', type=str, default=''
    )
    parser.add_argument(
        '--ngram_weight', type=float, default=0.9
    )
    parser.add_argument(
        '--n_layer', type=int, default=9
    )
    parser.add_argument(
        '--n_trial', type=int, default=100
    )
    args = parser.parse_args()
    return args


def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))


def rank_heads(head_scores, normalize='l2'):
    if normalize == 'l2':
        head_scores = (
            head_scores
            / head_scores.pow(2).sum(-1, keepdim=True).sqrt()
        )
    elif normalize == 'l1':
        head_scores = (
            head_scores
            / head_scores.abs().sum(-1, keepdim=True)
        )

    heads = []
    for layer in range(head_scores.shape[0]):
        for head in range(head_scores.shape[1]):
            heads.append(
                (layer, head, head_scores[layer, head].item())
            )

    return [
        (layer, head)
        for layer, head, _ in sorted(heads, key=lambda h: h[2], reverse=True)
    ]


def prune_head(model, retain_heads):
    heads_of_layers = [[] for _ in model.encoder.encoders]
    for layer, h in retain_heads:
        heads_of_layers[layer].append(h)

    for layer, heads in zip(model.encoder.encoders, heads_of_layers):
        n_head = layer.self_attn.h
        prune_qkv(layer.self_attn.linear_q, n_head, heads)
        prune_qkv(layer.self_attn.linear_k, n_head, heads)
        prune_qkv(layer.self_attn.linear_v, n_head, heads)
        prune_out(layer.self_attn.linear_out, n_head, heads)
        layer.self_attn.h = len(heads)


def prune_qkv(linear, n_head, retained_heads):
    assert linear.weight.shape[1] % n_head == 0
    head_dim = linear.weight.shape[1] // n_head
    linear.out_features = len(retained_heads) * head_dim
    new_weight = [
        linear.weight[i, :]
        for i in range(linear.weight.shape[0])
        if i // head_dim in retained_heads
    ]
    linear.weight.data = torch.stack(new_weight, 0)

    if linear.bias is not None:
        new_bias = [
            linear.bias[i]
            for i in range(linear.bias.shape[0])
            if i // head_dim in retained_heads
        ]
        linear.bias.data = torch.stack(new_bias, 0)


def prune_out(linear, n_head, retained_heads):
    assert linear.weight.shape[0] % n_head == 0
    head_dim = linear.weight.shape[0] // n_head
    linear.in_features = len(retained_heads) * head_dim
    new_weight = [
        linear.weight[:, i]
        for i in range(linear.weight.shape[1])
        if i // head_dim in retained_heads
    ]
    linear.weight.data = torch.stack(new_weight, 1)


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S'
    )
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main(args)

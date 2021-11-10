import torch
import argparse
import ipdb
import logging
import os
import sys
import math
import string
import soundfile
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from espnet2.bin.asr_inference import Speech2Text
from jiwer import wer


def main(args):
    random.seed(524)

    pt = torch.load(args.base_dir / 'head_grad.pt')
    head_scores = pt['accumulator']['encoder']
    if args.rank == 'importance':
        head_ranks = rank_heads(head_scores, normalize=args.normalize)
    else:
        head_ranks = [
            (layer, head)
            for layer in range(head_scores.shape[0])
            for head in range(head_scores.shape[1])
        ]

    avg_times = []
    avg_wers = []
    for ratio in [1 - i / 10 for i in range(0, 10)]:
        n_heads = [0] * head_scores.shape[0]
        n_retained = math.ceil(len(head_ranks) * ratio)
        retained_heads = set()
        for layer, head in head_ranks:
            if n_heads[layer] == 0:
                retained_heads.add((layer, head))
                n_heads[layer] += 1

        for layer, head in head_ranks:
            if len(retained_heads) < n_retained:
                retained_heads.add((layer, head))

        speech2text = Speech2Text.from_pretrained(
            args.pretrained_model_name,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=args.beam_size,
            ctc_weight=1.0,
            penalty=0.0,
            nbest=1,
            batch_size=1,
            lm_weight=args.lm_weight,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        prune_head(speech2text.asr_model, retained_heads)

        egs = pd.read_csv(f'{args.dir_librispeech}/test_clean.csv')
        refs = []
        hyps = []
        times = []

        rows = list(egs.iterrows())
        if args.n_sample is not None:
            rows = random.sample(rows, args.n_sample)

        for index, (_, row) in enumerate(tqdm(rows)):
            speech, rate = soundfile.read(args.dir_librispeech / row["path"])

            start = timer()
            nbests = speech2text(speech)
            end = timer()
            times.append(end - start)

            text, *_ = nbests[0]

            refs.append(text_normalizer(row['text']))
            hyps.append(text_normalizer(text))

        times = torch.tensor(times)
        er = wer(refs, hyps)
        print(f'Head ratio: {ratio}', flush=True)
        print(f'Time: {times.mean().item():.3f} ({times.std().item():.3f})', flush=True)
        print(f'WER: {er}', flush=True)
        avg_times.append(times.mean().item())
        avg_wers.append(er)

    print('Avg time:', avg_times)
    print('Avg WER:', avg_wers)


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
    parser.add_argument(
        '--dir_librispeech', type=Path,
        default=Path('~/espnet/data/librispeech/LibriSpeech/').expanduser()
    )
    parser.add_argument(
        '--n_sample', type=int, default=None
    )
    parser.add_argument(
        '--normalize', type=str, default='l2'
    )
    parser.add_argument(
        '--beam_size', type=int, default=5
    )
    parser.add_argument(
        '--lm_weight', type=float, default=1.0
    )
    parser.add_argument(
        '--rank', type=str, default='importance'
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

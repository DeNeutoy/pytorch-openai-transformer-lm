from typing import List
import re
import os
import time
import math
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial

from model_py import Model, LMHead, ClfHead, load_openai_pretrained_model
from text_utils import TextEncoder
from utils import (encode_dataset, flatten, iter_data,
                   ResultLogger, make_path)


def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    # input
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    # mask
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def create_batch_from_ids(sentences: List[List[int]],
                          max_timesteps: int,
                          total_vocab_size: int):
    batch_size = len(sentences)
    batch_tensor = np.zeros([batch_size, max_timesteps, 2], dtype=np.int32)
    mask = np.zeros([batch_size, max_timesteps], dtype=np.int32)

    for i, sentence in enumerate(sentences):
        if len(sentence) > max_timesteps:
            sentence = sentence[:max_timesteps]
        batch_tensor[i, :len(sentence), 0] = sentence
        mask[i, :len(sentence)] = 1

    batch_tensor[:, :, 1] = np.arange(total_vocab_size, total_vocab_size + max_timesteps)
    return batch_tensor, mask


def encode_sentences(sentences: List[List[str]], encoder):
    return [encoder.encode_sentence(sentence) for sentence in sentences]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)#4) # TODO add mutli-gpu training logic
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    import torch
    import numpy
    import random

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # torch.device object used throughout this script TODO add gpu setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    sent = encode_sentences([["this", "is", "a", "test"]], text_encoder)

    print(create_batch_from_ids(sent, 512, len(encoder))[0].shape)

    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = args.n_ctx//2-2
    vocab = n_vocab + n_special + args.n_ctx


    model = Model(args, vocab, args.n_ctx)
    load_openai_pretrained_model(model, n_ctx=args.n_ctx, n_special=n_special)
    model.to(device)


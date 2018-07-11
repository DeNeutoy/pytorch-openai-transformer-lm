from typing import List
import re
import os
import time
import math
import json
import random
import argparse
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial

from model_py import Model, LMHead, ClfHead, load_openai_pretrained_model
from text_utils import TextEncoder
from utils import (encode_dataset, flatten, iter_data,
                   ResultLogger, make_path)


def create_batch_from_ids(sentences: List[List[int]],
                          max_timesteps: int,
                          total_vocab_size: int):
    # TODO: do we need the delimiters? I think maybe not.
    batch_size = len(sentences)
    batch_tensor = np.zeros([batch_size, max_timesteps, 2], dtype=np.int32)
    mask = np.zeros([batch_size, max_timesteps], dtype=np.int32)

    for i, sentence in enumerate(sentences):
        if len(sentence) > max_timesteps:
            sentence = sentence[:max_timesteps]
        batch_tensor[i, :len(sentence), 0] = sentence
        mask[i, :len(sentence)] = 1

    # this adds the positional embedding on to each input.
    batch_tensor[:, :, 1] = np.arange(total_vocab_size, total_vocab_size + max_timesteps)
    return batch_tensor, mask


def encode_sentences(sentences: List[List[str]], encoder):

    encodings = []
    offsets = []
    for sent in sentences:
        encoding, offset = encoder.encode_sentence(sent)
        encodings.append(encoding)
        offsets.append(offset)
    return encodings, offsets

def get_word_embeddings(sentence_batch: torch.Tensor,
                        offsets: List[List[int]]):
    embeddings = []
    for batch, offset in zip(sentence_batch, offsets):
        word_representations = torch.index_select(batch, 1, offset)
        embeddings.append(word_representations.data.numpy())

    return embeddings


def tokenize(line):
    tokens = line.strip().split()
    if tokens[-1] in ('/.', '/?', '/!'):
        tokens[-1] = tokens[-1][1:]
    return tokens


def dump_openai_embeddings(datadir: str, text_encoder: TextEncoder, model, device):
    t1 = time.time()
    fname_in = os.path.join(datadir, 'sentences.txt')
    fname_ids = os.path.join(datadir, 'sentence_ids.txt')
    fname_out = os.path.join(datadir, 'openai_transformer.hdf5')
    with open(fname_in, 'r') as fin, \
         open(fname_ids, 'r') as fids, \
         h5py.File(fname_out, 'w') as fout:

        for ii, (sid, line) in enumerate(zip(fids, fin)):
            tokens = tokenize(line)
            encoded_sentence, offsets = encode_sentences([tokens], text_encoder)
            sentence_tensor, mask = create_batch_from_ids(encoded_sentence, 512, len(text_encoder.encoder))
            sentence_tensor= torch.from_numpy(sentence_tensor).long().to(device)
            embeddings = get_word_embeddings(model(sentence_tensor).cpu(), torch.tensor(offsets).long())
            embeds = np.asarray(embeddings)
            ds = fout.create_dataset(
                '{}'.format(sid.strip()),
                data=embeds,
                dtype=np.float32)
            if ii % 100 == 0:
                print(ii, time.time() - t1)


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
    parser.add_argument('--n_gpu', type=int, default=1)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = args.n_ctx//2-2
    vocab = n_vocab + n_special + args.n_ctx
    
    print("loading model")
    model = Model(args, vocab, args.n_ctx)
    load_openai_pretrained_model(model, n_ctx=args.n_ctx, n_special=n_special)
    print("moving to GPU")
    model.to(device)

    print("beginning embedding dump.")
    dump_openai_embeddings("/net/nfs.corp/allennlp/data/srl/test/test/", text_encoder, model, device)


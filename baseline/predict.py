import argparse
import numpy as np
import pandas as pd
import json
from collections import OrderedDict
from pprint import pprint

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from common.training_helpers import run_inference, default_prepare_for_training
from common.dataset import DfDataset
from baseline.data_helpers import load_vocab, create_line_encode_fn, create_open_fn
from baseline.train import prepare_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, default="./hparams.json")
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '-j', '--workers', default=None, type=int, metavar='N',
        help='number of data loading workers (default: 4)')
    parser.add_argument(
        '-b', '--batch-size', default=None, type=int,
        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument('--datapath', type=str, default=None)
    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--max-text-len', type=int, default=None)

    parser.add_argument('--test-csv', type=str)
    parser.add_argument('--out-csv', type=str)

    return parser.parse_args()


def prepare_data_pipeline(args):
    df = pd.read_csv(args.test_csv)
    df_dataset = list(df.to_dict("index").values())
    w2id, _ = load_vocab(args.vocab_path)
    line_encode = create_line_encode_fn(w2id, args.max_text_len)
    open_fn = create_open_fn(args.datapath, line_encode)

    val_dataset = DfDataset(
        df_dataset, open_fn,
        dict_transform=None,
        cache_prob=-1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        sampler=None)

    return df, val_loader


def batch_handler(dict_, model, logger, args):
    dict_.pop("target", None)

    input_var = {
        key: torch.autograd.Variable(
            value.cuda(async=True),
            volatile=True, requires_grad=False).detach()
        for key, value in dict_.items()}

    output = model(input_var).detach().data.cpu().numpy()

    return output


def main(args):
    hparams, _, _ = default_prepare_for_training(args)
    model, _, _, _ = prepare_model(args, hparams)
    pprint(model)

    df, loader = prepare_data_pipeline(args)

    predictions = run_inference(args, model, loader, batch_handler)
    predictions = np.concatenate(predictions, axis=0)
    df['label'] = predictions
    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)

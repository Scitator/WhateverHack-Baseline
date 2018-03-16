import argparse
import os
import pandas as pd
import torch
from torchnet import meter

from common.training_helpers import default_args, \
    default_prepare_for_training, \
    default_prepare_model, \
    default_prepare_data_pipeline, run_train
from common.dataset import column_fold_split

from baseline.data_helpers import load_vocab, create_line_encode_fn, create_open_fn
from baseline.model import build_baseline_model as BaselineModel


NETWORKS = {
    "baseline": BaselineModel
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser = default_args(parser)

    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--max-text-len', type=int, default=None)

    return parser.parse_args()


def prepare_model(args, hparams):
    model, criterion, optimizer, scheduler = default_prepare_model(
        args,
        hparams=hparams,
        available_networks=NETWORKS)

    return model, criterion, optimizer, scheduler


def prepare_data_pipeline(args):
    df = pd.read_csv(args.train_csv)
    w2id, _ = load_vocab(args.vocab_path)
    line_encode = create_line_encode_fn(w2id, args.max_text_len)
    open_fn = create_open_fn(args.datapath, line_encode)

    df = column_fold_split(
        df, column="image_name",
        folds_seed=args.folds_seed, n_folds=args.n_folds)
    df.to_csv(os.path.join(args.logdir, "datasplit.csv"))

    train_folds = list(map(int, args.train_folds.split(",")))
    df_train = df[df["fold"].isin(train_folds)]
    df_val = df[~df["fold"].isin(train_folds)]

    df_train = df_train.reset_index().drop("index", axis=1)
    df_val = df_val.reset_index().drop("index", axis=1)

    return default_prepare_data_pipeline(
        args, df_train, df_val,
        open_fn=open_fn)


def batch_handler(mode, epoch_metrics, dict_, model, criterion, optimizer, logger, args):
    if len(epoch_metrics) == 0:  # first time metric for epoch, initialization
        epoch_metrics = {
            "loss": meter.AverageValueMeter(),
            "auc": meter.AUCMeter()
        }

    target = dict_.pop("target")
    bs = len(target)

    if torch.cuda.is_available():
        input_var = {
            key: torch.autograd.Variable(value.cuda(async=True), requires_grad=False)
            for key, value in dict_.items()}
    else:
        input_var = {
            key: torch.autograd.Variable(value, requires_grad=False)
            for key, value in dict_.items()}

    if torch.cuda.is_available():
        target = target.cuda(async=True)
    target_var = torch.autograd.Variable(target, requires_grad=False)

    # compute output
    output = model(input_var)
    # @TODO: BCE loss issue
    output = output.squeeze(1)

    loss = criterion(output, target_var)
    loss_ = float(loss.data.cpu().squeeze().numpy()[0])
    epoch_metrics["loss"].add(loss_)
    logger.add_scalar("loss", loss_, args.step)

    epoch_metrics["auc"].add(output.data, target)
    logger.add_scalar("auc", epoch_metrics["auc"].value()[0], args.step)

    if mode == "train":
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_metrics["batch_size"] = bs

    return epoch_metrics


def main(args):
    run_train(
        args,
        prepare_for_training=default_prepare_for_training,
        prepare_model=prepare_model,
        prepare_data_pipeline=prepare_data_pipeline,
        batch_handler=batch_handler)


if __name__ == '__main__':
    args = parse_args()
    main(args)
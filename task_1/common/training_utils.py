import time
import numpy as np
import tqdm
import shutil
import torch


def get_val_from_metric(metric_value):
    if isinstance(metric_value, (int, float)):
        return metric_value
    else:
        metric_value = metric_value.value()
        if isinstance(metric_value, tuple):
            metric_value = metric_value[0]
        return metric_value


def run_train_val_loader(
        *, epoch, mode,
        loader, model, criterion, optimizer,
        batch_handler, logger, args):
    epoch_metrics = {}
    if mode == "train":
        # switch to train mode
        model.train()
    else:
        # switch to inference mode
        model.eval()

    end = time.time()
    step = epoch * len(loader) * args.batch_size
    for i, dict_ in enumerate(loader):
        logger.add_scalar("data time", time.time() - end, step)

        args.epoch, args.i, args.step = epoch, i, step

        epoch_metrics = batch_handler(
            mode, epoch_metrics, dict_, model, criterion, optimizer, logger, args)
        bs = epoch_metrics.get("batch_size", args.batch_size)
        epoch_metrics.pop("batch_size")

        # measure elapsed time
        elapsed_time = time.time() - end
        logger.add_scalar("batch time", elapsed_time, step)
        logger.add_scalar("sample per second", bs / elapsed_time, step)
        end = time.time()

        step += bs

    out_metrics = {key: get_val_from_metric(value) for key, value in epoch_metrics.items()}
    for key, value in out_metrics.items():
        logger.add_scalar("epoch " + key, value, epoch)
    epoch_metrics_str = "\t".join([
        "{key} {value:.4f}".format(key=key, value=value)
        for key, value in sorted(out_metrics.items())])
    print("{epoch} * Epoch ({mode}): ".format(epoch=epoch, mode=mode) + epoch_metrics_str)

    return out_metrics


def run_eval_loader(*,
        loader, model, batch_handler, logger, args):
    # switch to inference mode
    model.eval()

    predictions = []
    for dict_ in tqdm.tqdm(loader):
        predictions.append(batch_handler(dict_, model, logger, args))

    return predictions


def save_checkpoint(state, is_best, logdir):
    filename = "{}/checkpoint.pth.tar".format(logdir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/checkpoint.best.pth.tar'.format(logdir))

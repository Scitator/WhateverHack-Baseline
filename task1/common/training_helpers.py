import json
import os
import sys
from collections import OrderedDict
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from common.misc_utils import create_if_need, stream_tee, boolean_flag
from common.dataset import DfDataset
import common.training_utils as utils


def default_args(parser):
    parser.add_argument('--hparams', type=str, default="./hparams.json")
    parser.add_argument('--logdir', type=str, required=True)
    boolean_flag(parser, 'cuda', default=True, help='Use cuda to train model')

    parser.add_argument('--train-csv', type=str)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--n-folds', type=int, default=None)
    parser.add_argument('--folds-seed', type=int, default=42)
    parser.add_argument('--train-folds', type=str)
    parser.add_argument('--dataset-cache-prob', type=float, default=None)

    parser.add_argument(
        '-j', '--workers', default=None, type=int, metavar='N',
        help='number of data loading workers (default: 4)')
    parser.add_argument(
        '-b', '--batch-size', default=None, type=int,
        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument(
        '--epochs', default=None, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')

    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')

    # distributed training
    # parser.add_argument(
    #     '--world-size', default=1, type=int,
    #     help='number of distributed processes')
    # parser.add_argument(
    #     '--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #     help='url used to set up distributed training')
    # parser.add_argument(
    #     '--dist-backend', default='gloo', type=str,
    #     help='distributed backend')

    return parser


def default_prepare_for_training(args):
    train_logger, validation_logger = None, None

    if hasattr(args, "logdir"):
        create_if_need(args.logdir)
        train_log_dir = os.path.join(args.logdir, "train_log")
        validation_log_dir = os.path.join(args.logdir, "validation_log")

        logfile = open("{logdir}/log.txt".format(logdir=args.logdir), "w+")
        sys.stdout = stream_tee(sys.stdout, logfile)

        train_logger = SummaryWriter(train_log_dir)
        validation_logger = SummaryWriter(validation_log_dir)

    # load params
    with open(args.hparams, "r") as fin:
        hparams = json.load(fin, object_pairs_hook=OrderedDict)

    if hasattr(args, "logdir"):
        with open("{}/hparams.json".format(args.logdir), "w") as fout:
            json.dump(hparams, fout, indent=2)

    # hack with argparse in json
    training_args = hparams.pop("training_params", None)
    if training_args is not None:
        for key, value in training_args.items():
            arg_value = getattr(args, key, None)
            if arg_value is None:
                arg_value = value
            setattr(args, key, arg_value)

    return hparams, train_logger, validation_logger


def default_prepare_model(args, hparams, available_networks):
    # create model
    model_params = hparams["model_params"]
    model_name = model_params.pop("model", None)
    model = available_networks[model_name](**model_params)

    # define loss function (criterion), optimizer and scheduler
    criterion, optimizer, scheduler = None, None, None

    criterion_params = hparams.get("criterion_params", None) or {}
    criterion_name = criterion_params.pop("criterion", None)
    if criterion_name is not None:
        criterion = nn.__dict__[criterion_name](**criterion_params)
        if torch.cuda.is_available():
            criterion = criterion.cuda()

    optimizer_params = hparams.get("optimizer_params", None) or {}
    optimizer_name = optimizer_params.pop("optimizer", None)
    if optimizer_name is not None:
        optimizer = torch.optim.__dict__[optimizer_name](
            filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_params)

    scheduler_params = hparams.get("scheduler_params", None) or {}
    scheduler_name = scheduler_params.pop("scheduler", None)
    if scheduler_name is not None:
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](optimizer, **scheduler_params)

    return model, criterion, optimizer, scheduler


def default_prepare_data_pipeline(
        args, df_train, df_val, open_fn,
        train_dict_transforms=None,
        validaton_dict_transforms=None):
    df_train = list(df_train.to_dict("index").values())
    df_val = list(df_val.to_dict("index").values())

    train_dataset = DfDataset(
        df_train, open_fn=open_fn,
        dict_transform=train_dict_transforms,
        cache_prob=args.dataset_cache_prob)

    # if args.distributed:
    #     distributed_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    distributed_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(distributed_sampler is None),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        sampler=distributed_sampler)

    val_dataset = DfDataset(
        df_val, open_fn,
        dict_transform=validaton_dict_transforms,
        cache_prob=args.dataset_cache_prob)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        sampler=None)

    print("Train samples", len(train_loader))
    print("Val samples", len(val_loader))

    return train_loader, val_loader, distributed_sampler


def run_train(
        args, prepare_for_training, prepare_model, prepare_data_pipeline,
        batch_handler):
    hparams, train_logger, validation_logger = prepare_for_training(args)
    model, criterion, optimizer, scheduler = prepare_model(args, hparams)

    # load checkpoint
    best_loss = int(1e10)
    best_metrics = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            best_metrics = checkpoint['best_metrics']

            # model.load(checkpoint['model'])
            # optimizer.load(checkpoint['optimizer'])

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # debug
    pprint(model)
    pprint(criterion)
    pprint(optimizer)
    pprint(scheduler)
    pprint(args)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        model = torch.nn.DataParallel(model).cuda()
        # speed up
        cudnn.benchmark = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    train_loader, val_loader, distributed_sampler = prepare_data_pipeline(args)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     distributed_sampler.set_epoch(epoch)

        train_logger.add_scalar("epoch LR", scheduler.get_lr()[0], epoch)

        # train for one epoch
        utils.run_train_val_loader(
            epoch=epoch, mode="train",
            loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
            batch_handler=batch_handler, logger=train_logger, args=args)

        # evaluate on validation set
        epoch_val_metrics = utils.run_train_val_loader(
            epoch=epoch, mode="valid",
            loader=val_loader, model=model, criterion=criterion, optimizer=optimizer,
            batch_handler=batch_handler, logger=validation_logger, args=args)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_metrics["loss"])
        else:
            scheduler.step()

        # remember best loss and save checkpoint
        is_best = epoch_val_metrics["loss"] < best_loss
        best_loss = min(epoch_val_metrics["loss"], best_loss)
        best_metrics = epoch_val_metrics if is_best else best_metrics
        best_metrics = {
            key: value for key, value in best_metrics.items()
            if isinstance(value, float)}
        utils.save_checkpoint({
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "best_metrics": epoch_val_metrics,
            "model": model.module,
            "model_state_dict": model.module.state_dict(),
            "optimizer": optimizer,
            "optimizer_state_dict": optimizer.state_dict(),
        }, is_best, logdir=args.logdir)


def run_inference(
        args, model, loader, batch_handler):
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load(checkpoint['model'])

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        model = torch.nn.DataParallel(model).cuda()
        # speed up
        cudnn.benchmark = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    pprint(model)

    predictions = utils.run_eval_loader(
        loader=loader, model=model, batch_handler=batch_handler,
        logger=None, args=args)

    return predictions

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from pprint import pprint
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
import src.loader as loader
from src import Bar
import src.utils as utils
import src.misc as misc

from src.model.model import LinearModel, weight_init
import src.log as log
from src.dataset.datasets_basic import Human36M


stat_3d = torch.load('./data/stat_3d.pth.tar')
std_3d = Variable(torch.from_numpy(stat_3d['std'][stat_3d['dim_use']]).float().cuda())
mean_3d = Variable(torch.from_numpy(stat_3d['mean'][stat_3d['dim_use']]).float().cuda())


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to train (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
    if opt.is_train:
        test_loader = DataLoader(
            dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_sh, is_train=False),
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
        # loss_test, err_test = test(test_loader, model, criterion)
        # os._exit(0)
        train_loader = DataLoader(
            dataset=Human36M(actions=actions, data_path=opt.data_dir, use_hg=opt.use_sh),
            batch_size=opt.train_batch,
            shuffle=True,
            num_workers=opt.job,
            pin_memory=True)
    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma)
        loss_test, err_test = test(test_loader, model, criterion)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'flaot', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()


def test(test_loader, model, criterion):
    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', max=len(test_loader))

    for i, (inps, tars) in enumerate(test_loader):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        outputs_coord = outputs * std_3d + mean_3d
        loss = criterion(outputs_coord, targets)

        losses.update(loss.data[0], inputs.size(0))

        tars = (targets - mean_3d) / std_3d

        # calculate erruracy
        targets_unnorm = loader.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_unnorm = loader.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use]

        # for ba in range(inps.size(0)):
        #     gt = targets_use[ba].reshape(-1, 3)
        #     out = outputs_use[ba].reshape(-1, 3)
        #     _, Z, T, b, c = get_transformation(gt, out, True)
        #     out = (b * out.dot(T)) + c
        #
        #     outputs_use[ba, :] = out.reshape(1, 51)

        sqerr = (outputs_use - targets_use) ** 2

        distance = np.zeros((sqerr.shape[0], 17))
        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    main(option)

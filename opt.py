#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint

__all__ = ['Options']

actions = ["all",
           "All",
           "Directions",
           "Discussion",
           "Eating",
           "Greeting",
           "Phoning",
           "Photo",
           "Posing",
           "Purchases",
           "Sitting",
           "SittingDown",
           "Smoking",
           "Waiting",
           "WalkDog",
           "Walking",
           "WalkTogether"]


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir',       type=str, default='data/', help='path to dataset')
        self.parser.add_argument('--exp',            type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt',           type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--load',           type=str, default='', help='path to load a pretrained checkpoint')

        self.parser.add_argument('--test',           dest='test', action='store_true', help='test')
        self.parser.add_argument('--resume',         dest='resume', action='store_true', help='resume to train')

        self.parser.add_argument('--action',         type=str, default='All', choices=actions, help='All for all actions')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm',       dest='max_norm', action='store_true', help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size',    type=int, default=1024, help='size of each model layer')
        self.parser.add_argument('--num_stage',      type=int, default=2, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--use_hg',         dest='use_hg', action='store_true', help='whether use 2d pose from hourglass')
        self.parser.add_argument('--lr',             type=float,  default=1.0e-3)
        self.parser.add_argument('--lr_decay',       type=int,    default=100000, help='# steps of lr decay')
        self.parser.add_argument('--lr_gamma',       type=float,  default=0.96)
        self.parser.add_argument('--epochs',         type=int,    default=200)
        self.parser.add_argument('--dropout',        type=float,  default=0.5, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch',    type=int,    default=64)
        self.parser.add_argument('--test_batch',     type=int,    default=64)
        self.parser.add_argument('--job',            type=int,    default=8, help='# subprocesses to use for data loading')
        self.parser.add_argument('--no_max',         dest='max_norm', action='store_false', help='if use max_norm clip on grad')
        self.parser.add_argument('--max',            dest='max_norm', action='store_true', help='if use max_norm clip on grad')
        self.parser.set_defaults(max_norm=True)
        self.parser.add_argument('--procrustes',     dest='procrustes', action='store_true', help='use procrustes analysis at testing')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        if self.opt.load:
            if not os.path.isfile(self.opt.load):
                print ("{} is not found".format(self.opt.load))
        self.opt.is_train = False if self.opt.test else True
        self.opt.ckpt = ckpt
        self._print()
        return self.opt

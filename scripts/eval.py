#coding=utf-8

# simple script to start a multi-gpu evaluating process on brainpp workspace
# usage: python3 scripts/eval.py configs/....

import sys
import os
import argparse

parser = argparse.ArgumentParser('Eval')

parser.add_argument('cfg',
                    type=str)

parser.add_argument('-ngpu',
                    type=int,
                    default=4)

parser.add_argument('-ncpu',
                    required=False,
                    type=int,
                    default=8)

parser.add_argument('-mem',
                    required=False,
                    type=int,
                    default=40)

parser.add_argument('-p',
                    action='store_true',
                    default=False)

args = parser.parse_args()

NGPU, NCPU, MEM, CFG = args.ngpu, args.ncpu, args.mem * 1024, args.cfg

if args.p:
    code = 'rlaunch --gpu=%d --cpu=%d --memory=%d --preemptible yes --' \
           ' python3 -m torch.distributed.launch' \
           ' --nproc_per_node=%d tools/test_net.py' \
           ' --config-file ' \
           '%s' % (NGPU, NCPU, MEM, NGPU, CFG)
else:
    code = 'rlaunch --gpu=%d --cpu=%d --memory=%d --' \
           ' python3 -m torch.distributed.launch' \
           ' --nproc_per_node=%d tools/test_net.py' \
           ' --config-file ' \
           '%s' %(NGPU, NCPU, MEM, NGPU, CFG)


os.system(code)
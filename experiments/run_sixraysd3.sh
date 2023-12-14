#!/bin/bash

python train.py --dataset sixray_sd3                \
--isize 128 --niter 100 --model ganomaly        \
--name sixray_sd3_experiment\
--batchsize 32 --dataroot /data/sixray_sd3_anomaly --display
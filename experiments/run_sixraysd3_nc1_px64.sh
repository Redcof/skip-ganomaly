#!/bin/bash

python train.py --dataset sixray_sd3                  \
--isize 64 --niter 50 --manualseed 47 --model ganomaly              \
--name sixray_sd3_grey_exp_px64     --nc 1          \
--batchsize 32 --dataroot /data/sixray_sd3_anomaly --display
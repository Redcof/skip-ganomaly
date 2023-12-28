#!/bin/bash

python train.py --dataset sixray_sd3                  \
--isize 128 --niter 50  --manualseed 47 --model ganomaly              \
--name sixray_sd3_grey_experiment     --nc 1          \
--batchsize 32 --dataroot /data/sixray_sd3_anomaly --display
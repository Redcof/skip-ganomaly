#!/bin/bash

python train.py --dataset sixray_sd3                  \
--isize 64 --niter 50 --manualseed 47 --model ganomaly              \
--name sixraysd3_nc1_px64_traycrop_gun     --nc 1          \
--batchsize 32 --dataroot /data/sixray_sd3_anomaly --display
#!/bin/bash

python train.py --dataset sixray_sd3                  \
--isize 64 --niter 75 --manualseed 47 --model ganomaly              \
--name exp5_sixraysd3_ep75_nc1_px64_traycrop_gun     --nc 1          \
--batchsize 32 --dataroot /data/sixray_sd3_anomaly --display
"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""
import torch

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model


##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    torch.autograd.set_detect_anomaly(True)
    model.train()


if __name__ == '__main__':
    main()

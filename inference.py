"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run inference.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --model-path <path/to/checkpoint/dir>       \
"""

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
    model.train()


if __name__ == '__main__':
    main()

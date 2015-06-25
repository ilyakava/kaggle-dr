import argparse

def get():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n",
                        "--network",
                        type=str,
                        default="vgg_mini7b_leak")
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        default="data/train/centered_crop/")
    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=128)
    parser.add_argument("-c",
                        "--center",
                        type=int,
                        default=1,
                        help="Sumtract mean example from examples.")
    parser.add_argument("-z",
                        "--normalize",
                        type=int,
                        default=1,
                        help="Divide examples by std dev of examples.")
    parser.add_argument("-l",
                        "--learning-rate",
                        type=float,
                        default=0.01,
                        help="Initial learning rate.")
    parser.add_argument("-m",
                        "--momentum",
                        type=float,
                        default=0.9)
    parser.add_argument("-a",
                        "--alpha",
                        type=int,
                        default=3,
                        help="Inverse of this number will be the degree of leakiness for LReLU units if any.")
    parser.add_argument("-x",
                        "--max-epochs",
                        type=int,
                        default=60)
    parser.add_argument("-A",
                        "--amplify",
                        type=int,
                        default=1,
                        help="Factor by which each input example will be scaled immediately before running network.")
    parser.add_argument("-o",
                        "--output-classes",
                        type=int,
                        default=5,
                        help="num_units in the network OUTPUT layer.")
    parser.add_argument("-p",
                        "--decay-patience",
                        type=int,
                        default=9999,
                        help="Number epochs of worse than best validation performance to wait before decaying leaning rate by --decay-factor")
    parser.add_argument("-f",
                        "--decay-factor",
                        type=int,
                        default=2,
                        help="Number with which to divided learning rate after --decay-patience is passed.")
    parser.add_argument("-L",
                        "--loss-type",
                        type=str,
                        default="one-hot")
    parser.add_argument("-P",
                        "--validations-per-epoch",
                        type=int,
                        default=1,
                        help="Number of times to validate and print confusion matrix per epoch.")
    parser.add_argument("-g",
                        "--as-grey",
                        type=int,
                        default=1,
                        help="1 for grayscale, 0 for rgb")

    return parser.parse_args()
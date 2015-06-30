import argparse

def get():
    parser = argparse.ArgumentParser()

    parser.add_argument("-M",
                        "--model-file",
                        type=str,
                        default=None)
    parser.add_argument("-d",
                        "--train-dataset",
                        type=str,
                        default="data/train/centered_crop/",
                        help="This is needed to calculate the centering and standardization images to subtract and divide the test examples with.")
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
    parser.add_argument("-F",
                        "--train-flip",
                        type=str,
                        default='no_flip',
                        help="Method name or csv file that contains complete information on whether to flip a given training image.")
    parser.add_argument("-D",
                        "--test-dataset",
                        type=str,
                        default=None)
    parser.add_argument("-r",
                        "--random-seed",
                        type=int,
                        default=1991,
                        help="Make validation set selection reproducible")
    parser.add_argument("-v",
                        "--valid-dataset-size",
                        type=int,
                        default=1664,
                        help="Validation set size (4864=14%, 3456=10%, 1664=5%)")

    return parser.parse_args()
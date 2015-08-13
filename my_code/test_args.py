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
    parser.add_argument("-V",
                        "--train-labels-csv-path",
                        type=str,
                        default="data/train/trainLabels.csv",
                        help="This is needed b/c it acts as a list of what images are in the training set.")
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
    parser.add_argument("-F3",
                        "--test-flip",
                        type=str,
                        default='no_flip')
    parser.add_argument("-D",
                        "--test-dataset",
                        type=str,
                        default=None,
                        help="A directory with test images.")
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
    parser.add_argument("-fs",
                        "--filter-shape",
                        type=str,
                        default='c01b',
                        choices=['c01b', 'bc01'],
                        help="The shape of the filters in the CONV layer. Use 'bc01' to use slower shape (this option exists to run legacy models trained in the suboptimal shape). You must use 'bc01' if you are not using cuda_convnet.")
    parser.add_argument("-cc",
                        "--cuda-convnet",
                        type=int,
                        default=1,
                        choices=[0,1],
                        help="If you do not have a GPU, you must pass '-cc 0' (and don't forget to set THEANO_FLAGS='device=cpu'). If 1: use cuda_convnet library for convolutions which requires a GPU. Else use theano defaults which work on CPU and GPU.")
    parser.add_argument("-H",
                        "--cache-size-factor",
                        type=int,
                        default=8,
                        help="The number of multiples of minimatches to store in GPU device memory at once.")

    return parser.parse_args()

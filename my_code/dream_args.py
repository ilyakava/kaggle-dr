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
                        default="data/train/trainLabels.csv")
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
                        "--test-imagepath",
                        type=str,
                        default=None,
                        help="Either a path to an image file (with extension) or directory of images.")
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
    parser.add_argument("-p",
                        "--patch-size",
                        type=int,
                        default=11)
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
    parser.add_argument("-i",
                        "--itr-per-octave",
                        type=int,
                        default=10)
    parser.add_argument("-s",
                        "--step-size",
                        type=float,
                        default=1.5)
    parser.add_argument("-o",
                        "--max-octaves",
                        type=int,
                        default=4)
    parser.add_argument("-S",
                        "--octave-scale",
                        type=float,
                        default=1.4)

    return parser.parse_args()
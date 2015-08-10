import argparse

LONG_TIME = 9999

def get():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n",
                        "--network",
                        type=str,
                        default="vgg_mini7b_leak_sig")
    parser.add_argument("-d",
                        "--train-dataset",
                        type=str,
                        default="data/train/centered_crop/")
    parser.add_argument("-V",
                        "--train-labels-csv-path",
                        type=str,
                        default="data/train/trainLabels.csv")
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
                        default=150)
    parser.add_argument("-A",
                        "--amplify",
                        type=int,
                        default=1,
                        help="Factor by which each input example will be scaled immediately before running network.")
    parser.add_argument("-o",
                        "--output-classes",
                        type=int,
                        default=4,
                        help="num_units in the network OUTPUT layer.")
    parser.add_argument("-p",
                        "--decay-patience",
                        type=int,
                        default=LONG_TIME,
                        help="Number epochs of worse than best validation performance to wait before decaying leaning rate by --decay-factor")
    parser.add_argument("-f",
                        "--decay-factor",
                        type=int,
                        default=2,
                        help="Number with which to divided learning rate after --decay-patience is passed.")
    parser.add_argument("-i",
                        "--decay-limit",
                        type=int,
                        default=10,
                        help="Maximum number of times to decay learning rate.")
    parser.add_argument("-L",
                        "--loss-type",
                        type=str,
                        default="nnrank-re-kappa-sym")
    parser.add_argument("-P",
                        "--validations-per-epoch",
                        type=float,
                        default=1,
                        help="Number of times to validate and print confusion matrix per epoch.")
    parser.add_argument("-g",
                        "--as-grey",
                        type=int,
                        default=0,
                        help="1 for grayscale, 0 for rgb")
    parser.add_argument("-F",
                        "--train-flip",
                        type=str,
                        default='rand_flip',
                        help="Method name or csv file (aligned flips) that contains complete information on whether to flip a given training image.")
    parser.add_argument("-F2",
                        "--valid-flip",
                        type=str,
                        default='no_flip')
    parser.add_argument("-F3",
                        "--test-flip",
                        type=str,
                        default='no_flip')
    parser.add_argument("-s",
                        "--shuffle",
                        type=int,
                        default=0,
                        help="1 to shuffle training set every epoch, 0 to use the same ordering")
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
                        default=4864,
                        help="Validation set size (4864=14%, 3456=10%, 1664=5%)")
    parser.add_argument("-t",
                        "--noise-decay-start",
                        type=int,
                        default=LONG_TIME,
                        help="This is relevant only when random flips should transition into aligned flips.")
    parser.add_argument("-u",
                        "--noise-decay-duration",
                        type=int,
                        default=50,
                        help="Pass 0 for this option (in conjunction with: '-F *.csv') for flips from csv with no noise. This is relevant only when random flips should transition into aligned flips.")
    parser.add_argument("-y",
                        "--noise-decay-severity",
                        type=float,
                        default=5,
                        help="e ** (n/ (--noise-decay-duration / --noise-decay-severity)) is the noise at n. This is relevant only when random flips should transition into aligned flips.")
    parser.add_argument("-S",
                        "--sample-class",
                        type=int,
                        default=None,
                        help="Instead of runing through all of the data in whatever distribution it lies in, make each batch have a custom (default: uniform) class distribution by running through a single class's data while over/undersampling other classes' data. For the following choices for '-U', multiply '-x' by: 0: 0.276,  1: 2.95,  2: 1.35 , 3: 8.07,  4: 9.80, to get the same runtime for unform distributions")
    parser.add_argument("-R",
                        "--custom-distribution",
                        nargs='+',
                        type=int,
                        default=None,
                        help="When used in conjunction with -S, changes the uniform distribution of classes to whatever ascending class frequency counts specified here. '-R 94 9 19 3 3' replicates the training set distribution. To set a new '-x', multiply the old '-x' by: ((274*custom_distribution) / [25810,2443,5292,873,708])[sample_class].")
    parser.add_argument("-C",
                        "--train-color-cast",
                        type=str,
                        default='baidu_cast',
                        help="Method that returns integers to add to image channels.")
    parser.add_argument("-C2",
                        "--valid-color-cast",
                        type=str,
                        default='no_cast')
    parser.add_argument("-C3",
                        "--test-color-cast",
                        type=str,
                        default='no_cast')
    parser.add_argument("-G",
                        "--color-cast-range",
                        type=int,
                        default=30,
                        help="Maximum value with which to bias a color channel.")
    parser.add_argument("-O",
                        "--override-input-size",
                        type=int,
                        default=None,
                        help="Override the size of the image used as reported in network_specs.json.")
    parser.add_argument("-M",
                        "--model-file",
                        type=str,
                        default=None)
    parser.add_argument("-fs",
                        "--filter-shape",
                        type=str,
                        default='c01b',
                        choices=['c01b', 'bc01'],
                        help="The shape of the filters in the CONV layer. Use 'bc01' to use slower shape (this option exists to run legacy models trained in the suboptimal shape). You must use 'bc01' if you are not using cuda_convnet.")
    parser.add_argument("-H",
                        "--cache-size-factor",
                        type=int,
                        default=8,
                        help="The number of multiples of minimatches to store in GPU device memory at once.")
    parser.add_argument("-cc",
                        "--cuda-convnet",
                        type=int,
                        default=1,
                        choices=[0,1],
                        help="If you do not have a GPU, you must pass '-cc 0' (and don't forget to set THEANO_FLAGS='device=cpu'). If 1: use cuda_convnet library for convolutions which requires a GPU. Else use theano defaults which work on CPU and GPU.")
    parser.add_argument("-k1",
                        "--pre-train-crop",
                        type=str,
                        default='center_crop',
                        help="Name of method that returns integers ranges to crop an image by.")
    parser.add_argument("-k2",
                        "--train-crop",
                        type=str,
                        default='center_crop')
    parser.add_argument("-k3",
                        "--valid-test-crop",
                        type=str,
                        default='center_crop')

    return parser.parse_args()
# Getting Started

## Code (tested on Ubuntu 14.10)

### Install for a new Machine

`sudo apt-get install -y git python-pip python-yaml python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose libfreetype6-dev libpng-dev`

### Install General Python deps:

`sudo pip install theano scikit-learn scikit-image nyanbar natsort`

If `skimage.io` has issues, try: `sudo pop install -U scikit-image`.

### Install Lasange

```
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne/
pip install -r requirements.txt
sudo python setup.py install
```

### Install pylearn2

```
git clone git://github.com/lisa-lab/pylearn2.git && cd pylearn2/ && sudo python setup.py develop
```

#### Create a [~/.theanorc](http://deeplearning.net/software/theano/library/config.html) file

Ex Contents:

```
[global]
floatX = float32
device = gpu0

```

[Override with another device via](http://deeplearning.net/software/theano/library/config.html): `THEANO_FLAGS='device=gpu0'` prefix. Get a list of gpus via: `nvidia-smi -L`.

### Clone Code from ciresan and theanet

`git submodule update --init --recursive`

## Data

### Install

`sudo apt-get install -y p7zip-full graphicsmagick`

### Download & Unpack Data (~45 mins for test)

Download from [kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?trainLabels.csv.zip) (maybe with w3m) and place in `data/train`, to unpack run:

`7z e -oorig/ train.zip.001`

This will place the images into `data/train/orig`

After placing the test zip files into `data/test` you can run a similar command `7z e -oorig/ test.zip.001` to place the images into `data/test/orig`

Place `trainLabels.csv` into `data/train`

### Preparing Data for the Network

#### Full Size Originals -> Smaller Originals (~2.5 images per second)

(Ex) This will create 3 batchfiles for graphicsmagick to output 128x128 pngs:

```
mkdir data/train/centered_crop
python my_code/create_resize_batchfiles.py data/train/orig/ data/train/centered_crop/ 2 128 3
```

Then follow the on screen directions, which will list what commands to run to process the images cataloged in the generated batchfiles.

Depending on how your CPU schedules, more than 1 batchfile may not result in any speedup (3 is the best size for me however).

#### Alignment (~3 seconds per image)

To reduce noise in the training dataset, detect which images are inverted (taken with an indirect ophthalmoscope) and which are left/right, and invert the images until optic nerve is on the right side of the image.

`python my_code/batch_align.py data/train/orig/ n i`

This will run the ith of n partitions that creates a csv of which inversions to perform on the images in that partition. For example, you could run:

`python my_code/batch_align.py data/train/orig/ 3 1`
`python my_code/batch_align.py data/train/orig/ 3 2`
`python my_code/batch_align.py data/train/orig/ 3 3`

In three different `screen` sessions for parallel processing. Each will report having created a csv file. You can join these multiple csvs into one with: `awk 'FNR==1 && NR!=1{next;}{print}' *.csv > my.csv`

## The Network

### Training the network

`python -m my_code.VGGNet`

### Testing a single network

`python -m my_code.predict`

# Running tests

`make test`

# Getting Help

- [SO](http://stackoverflow.com/questions/tagged/neural-network)
- [DataScience Beta](http://datascience.stackexchange.com/questions/tagged/deep-learning)
- [CrossValidated](http://stats.stackexchange.com/questions/tagged/deep-learning)

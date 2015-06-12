# Getting Started

## Code (tested on Ubuntu 14.10)

### Install for a new Machine

`sudo apt-get install -y git python-pip python-yaml python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose libfreetype6-dev libpng-dev`

### Install General Python deps:

`sudo pip install -U theano scikit-learn scikit-image`

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

### Download & Unpack Data

Download from [kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?trainLabels.csv.zip) (maybe with w3m), run to unpack:

`7z x train.zip.001`

Place these images into `data/train/orig`

Place `trainLabels.csv` into `data/`

### Preparing Data for the Network

#### Full Size Originals -> Smaller Originals (~2 hours)

Ex:

```
mkdir data/train/128_simple
python data/create_resize_batchfiles.py data/train/orig/ data/train/128_simple/ 2 128
```

#### Standardization

... Coming soon

## The Network

### Training the network

`python -m my_code.VGGNet`
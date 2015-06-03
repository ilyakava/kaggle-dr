## Getting Started

### Clone Code from ciresan and theanet

`git submodule update --init --recursive`

### Install Lasange

```
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne/
pip install -r requirements.txt
sudo python setup.py install
```

### Download & Unpack Data

Download from [kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data?trainLabels.csv.zip), run to unpack:

`7z x train.zip.001`

Place these images into `data/train/orig`

Place `trainLabels.csv` into `data/`

### Preparing Data for the Network

#### Full Size Originals -> Smaller Originals (~2 hours)

requires graphicsmagick

Ex:

```
mkdir data/train/simple_crop
python data/create_resize_batchfiles.py
```

#### Standardization

... Coming soon

#### Packaging for Network (~8 minutes)

```
python -m my_code.create_train_val_test_set
```

### Training the network

`python -m my_code.VGGNet`
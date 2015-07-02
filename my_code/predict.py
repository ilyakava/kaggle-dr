from os import path
import time
import cPickle
import csv
# this repo
from my_code.VGGNet import VGGNet
from my_code.data_stream import DataStream
import my_code.test_args as test_args

import pdb

def save_prediction(runid, img_names, labels):
    """
    Saves a csv for kaggle in the same format as the training Labels
    """
    assert(len(img_names) == len(labels))
    outfile = "results/%s.csv" % runid
    print("Writing results to %s" % outfile)
    with open(outfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image','level'])
        for i in xrange(len(img_names)):
            writer.writerow([img_names[i],labels[i]])

def model_runid(model_file):
    return path.splitext(path.basename(model_file))[0].split('-')[0]

def load_column(model_file, train_dataset, center, normalize, train_flip,
                test_dataset, random_seed, valid_dataset_size):
    print("Loading...")
    f = open(model_file)
    batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape = cPickle.load(f)
    f.close()

    data_stream = DataStream(train_image_dir=train_dataset, batch_size=batch_size, image_shape=image_shape, center=center, normalize=normalize, train_flip=train_flip, test_image_dir=test_dataset, random_seed=random_seed, valid_dataset_size=valid_dataset_size)
    column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape)
    column.restore(model_file)
    return column

def single(model_file, **kwargs):
    assert(model_file)
    runid = model_runid(model_file)

    column = load_column(model_file, **kwargs)
    try:
        all_test_predictions, all_test_output = column.test()
        save_prediction(runid, column.ds.test_dataset['X'], all_test_predictions)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Testing"
    print(time.strftime("Finished at %H:%M:%S on %Y-%m-%d"))

if __name__ == '__main__':
    _ = test_args.get()

    single(model_file=_.model_file,
           train_dataset=_.train_dataset,
           center=_.center,
           normalize=_.normalize,
           train_flip=_.train_flip,
           test_dataset=_.test_dataset,
           random_seed=_.random_seed,
           valid_dataset_size=_.valid_dataset_size)

from os import listdir, path
import re
import numpy
import cPickle

# import matplotlib
# matplotlib.use('Agg')
# from skimage.io import imread
# matplotlib.rcParams.update({'font.size': 2})
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

import theano

from my_code.VGGNet import VGGNet
from my_code.data_stream import DataStream

import my_code.dream_args as args
from my_code.predict import model_runid

import pdb

# class DreamStudy(object):
#     def __init__(self, data_stream, test_imagepath=None):
#         """

#         """
#         self.ds = data_stream
#         self.test_imagepath = test_imagepath

#     def buffer_dream_dataset(self):
#         assert(self.test_imagepath)
#         img = self.ds.feed_image(image_name=self.test_imagepath, image_dir='')
#         channel_means = img.mean(axis=(0,1))

#         x_cache_block = numpy.zeros(((self.ds.cache_size,) + self.ds.image_shape), dtype=theano.config.floatX)
#         n_full_cache_blocks, n_leftovers = divmod(self.num_patch_centers, self.ds.cache_size)
#         for ith_cache_block in xrange(n_full_cache_blocks):
#             ith_cache_block_end = (ith_cache_block + 1) * self.ds.cache_size
#             idxs_to_full_dataset = list(range(ith_cache_block * self.ds.cache_size, ith_cache_block_end))
#             for ci,n in enumerate(idxs_to_full_dataset):
#                 i_start,i_end,j_start,j_end = self.nth_patch(n)

#                 x_cache_block[ci, ...] = img
#                 x_cache_block[ci, i_start:i_end, j_start:j_end, :] = channel_means
#             yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')
#         # sneak the leftovers out, padded by the previous full cache block
#         if n_leftovers:
#             for ci, n in enumerate(list(xrange(ith_cache_block_end, len(self.patch_starts)))):
#                 i_start,i_end,j_start,j_end = self.nth_patch(n)

#                 x_cache_block[ci, ...] = img
#                 x_cache_block[ci, i_start:i_end, j_start:j_end, :] = channel_means
#                 idxs_to_full_dataset[ci] = n
#             yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')

#     def accumulate_patches_into_heatmaps(self, all_test_output, outpath_prefix=''):
#         outpath = "plots/%s_%s.png" % (outpath_prefix, path.splitext(path.basename(self.test_imagepath))[0])
#         # http://matplotlib.org/examples/axes_grid/demo_axes_grid.html
#         fig = plt.figure()
#         grid = AxesGrid(fig, 143, # similar to subplot(143)
#                     nrows_ncols = (1, 1))
#         orig_img = imread(self.test_imagepath+'.png')
#         grid[0].imshow(orig_img)
#         grid = AxesGrid(fig, 144, # similar to subplot(144)
#                     nrows_ncols = (2, 2),
#                     axes_pad = 0.15,
#                     label_mode = "1",
#                     share_all = True,
#                     cbar_location="right",
#                     cbar_mode="each",
#                     cbar_size="7%",
#                     cbar_pad="2%",
#                     )

#         for klass in xrange(all_test_output.shape[1]):
#             accumulator = numpy.zeros(self.ds.image_shape[:2])
#             normalizer = numpy.zeros(self.ds.image_shape[:2])
#             for n in xrange(self.num_patch_centers):
#                 i_start,i_end,j_start,j_end = self.nth_patch(n)

#                 accumulator[i_start:i_end, j_start:j_end] += all_test_output[n,klass]
#                 normalizer[i_start:i_end, j_start:j_end] += 1
#             normalized_img = accumulator / normalizer
#             im = grid[klass].imshow(normalized_img, interpolation="nearest", vmin=0, vmax=1)
#             grid.cbar_axes[klass].colorbar(im)
#         grid.axes_llc.set_xticks([])
#         grid.axes_llc.set_yticks([])
#         print("Saving figure as: %s" % outpath)
#         plt.savefig(outpath, dpi=600, bbox_inches='tight')

# def file_iter(test_path):
#     """
#     Iterates through full paths of images.
#     """
#     e = ValueError("'%s' is neither a file nor a directory of images" % test_path)
#     if path.isdir(test_path):
#         images = [path.splitext(f)[0] for f in listdir(test_path) if re.search('\.(jpeg|png)', f)]
#         if not len(images):
#             raise e
#         for image in images:
#             yield test_path + image
#     elif path.isfile(test_path):
#         yield path.splitext(test_path)[0]
#     else:
#         raise e

def load_column(model_file, batch_size, learning_rate, train_dataset, train_labels_csv_path, center, normalize, train_flip,
                test_dataset, random_seed, valid_dataset_size, filter_shape, cuda_convnet):
    print("Loading Model...")
    f = open(model_file)
    _batch_size, _init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape = cPickle.load(f)
    f.close()

    data_stream = DataStream(train_image_dir=train_dataset, train_labels_csv_path=train_labels_csv_path, image_shape=image_shape, batch_size=batch_size, cache_size_factor=1, center=center, normalize=normalize, train_flip=train_flip, test_image_dir=test_dataset, random_seed=random_seed, valid_dataset_size=valid_dataset_size)
    column = VGGNet(data_stream, batch_size, learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape, filter_shape, cuda_convnet)
    column.restore(model_file)
    return column

def plot_dreams(model_file, test_path, max_itr, **kwargs):
    assert(model_file)
    runid = model_runid(model_file)

    column = load_column(model_file, batch_size=1, learning_rate=1, **kwargs)

    try:
        itr = 0
        batch = numpy.zeros((1,) + column.ds.image_shape)
        batch[0] = column.ds.feed_image(image_name=test_path, image_dir='')
        reshaped_batch = numpy.rollaxis(batch, 3, 1)
        column.x_buffer.set_value(lasagne.utils.floatX(reshaped_batch), borrow=True)
        while itr < max_itr:

            learning_rate = 1
            batch_updates = self.dream_batch(learning_rate)
            reshaped_batch += batch_updates
            column.x_buffer.set_value(lasagne.utils.floatX(reshaped_batch), borrow=True)

            itr += 1
        pdb.set_trace()
        scipy.misc.toimage(layer_img).save('data/dreams/%i_itr.png' % itr, "PNG")
    except KeyboardInterrupt:
        print "[ERROR] User terminated Dream Study"
    print "Done"

if __name__ == '__main__':
    _ = args.get()

    plot_dreams(model_file=_.model_file,
           test_path=_.test_path,
           max_itr=_.max_itr,
           train_dataset=_.train_dataset,
           train_labels_csv_path=_.train_labels_csv_path,
           center=_.center,
           normalize=_.normalize,
           train_flip=_.train_flip,
           test_dataset=None,
           random_seed=_.random_seed,
           valid_dataset_size=_.valid_dataset_size,
           filter_shape=_.filter_shape,
           cuda_convnet=_.cuda_convnet)

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
import lasagne

from my_code.VGGNet import VGGNet
from my_code.data_stream import DataStream

import my_code.dream_args as args
from my_code.predict import model_runid

import scipy.misc

import pdb

def calculate_octave_and_tile_sizes(source_size, nn_image_size, max_octaves=4, octave_scale=1.4):
    # find octave sizes
    # array of [h,w] arrays
    octave_sizes = [list(source_size)]
    while len(octave_sizes) < max_octaves and min(octave_sizes[-1]) > nn_image_size:
        min_dim = min(octave_sizes[-1])
        scale = min(octave_scale, float(min_dim) / nn_image_size)

        new_dims = [int(dim / scale) for dim in octave_sizes[-1]]
        octave_sizes.append(new_dims)
    assert(numpy.array(octave_sizes).min() >= nn_image_size)

    # calculate tile limits per octave (and normalizing coefs)
    octave_tile_corners = []
    for size in octave_sizes:
        h,w = size

        n_minus1_tiles_h, nth_tile_offset_h = divmod(h, nn_image_size)
        tops = [nn_image_size * i for i in range(n_minus1_tiles_h)]
        tops.append(tops[-1]+nth_tile_offset_h)

        n_minus1_tiles_w, nth_tile_offset_w = divmod(w, nn_image_size)
        lefts = [nn_image_size * i for i in range(n_minus1_tiles_w)]
        lefts.append(lefts[-1]+nth_tile_offset_w)

        tile_corners = []
        for top in tops:
            for left in lefts:
                tile_corners.append([top,left])
        octave_tile_corners.append(tile_corners)
    return(octave_sizes,octave_tile_corners)

# class DreamStudyBuffer(object):
#     """
#     Keeps the state of the dream in a double buffer
#     (source->batch, batch_output->source, and repeat)
#     """

#     def __init__(self, data_stream, test_imagepath=None):
#         OCTAVE_SCALE = 1.4
#         MAX_OCTAVES = 4
#         NN_IMG_SIZE = data_stream.image_shape[0]

#         # input image
#         source = data_stream.feed_image(image_name=test_imagepath, image_dir='')



#         # resize source for octave images (another function) cur_octaves
#         # with another function: initialize self.batch by tiling self.cur_octaves

#         # --- write to source

#         # map back gradients to each octave and normalize

#         # enlarge each octave to original image size

#         # update source image

#         # --- reset batch from new source

#         # reset batch from new source image

#         self.octave_images
#         self.octave_tiles


#         assert(occlusion_patch_size % 2 == 1) # for simplicity
#         self.ds = data_stream
#         self.occlusion_patch_size = occlusion_patch_size
#         self.test_imagepath = test_imagepath

#         patched_img_pad = (occlusion_patch_size - 1) / 2
#         patched_img_dim = data_stream.image_shape[0] - patched_img_pad*2
#         self.num_patch_centers = (patched_img_dim)**2 # number of images in test set

#         self.patch_starts = [divmod(n, patched_img_dim) for n in xrange(self.num_patch_centers)]

#     def nth_patch(self, n):
#         i_start,j_start = self.patch_starts[n]
#         i_end = i_start + self.occlusion_patch_size
#         j_end = j_start + self.occlusion_patch_size

#         return(i_start,i_end,j_start,j_end)

#     def buffer_occluded_dataset(self):
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

# Layers to choose:

# 1: ShuffleBC01ToC01BLayer
# 2: Conv2DCCLayer
# 3: MaxPool2DCCLayer
# 4: DropoutLayer
# 5: Conv2DCCLayer
# 6: MaxPool2DCCLayer
# 7: DropoutLayer
# 8: Conv2DCCLayer
# 9: DropoutLayer
# 10: Conv2DCCLayer
# 11: MaxPool2DCCLayer
# 12: DropoutLayer
# 13: Conv2DCCLayer
# 14: MaxPool2DCCLayer
# 15: DropoutLayer
# 16: Conv2DCCLayer
# 17: MaxPool2DCCLayer
# 18: ShuffleC01BToBC01Layer
# 19: DropoutLayer
# 20: DenseLayer
# 21: FeaturePoolLayer
# 22: DropoutLayer
# 23: DenseLayer
# 24: FeaturePoolLayer
# 25: DropoutLayer
# 26: DenseLayer

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

        while itr <= max_itr:
            if (itr in set([0] + [int(i) for i in numpy.logspace(0,numpy.log10(max_itr),10)])):
                name = 'data/dreams/%i_itr.png' % itr
                print("saving %s" % name)
                scipy.misc.toimage(numpy.rollaxis(reshaped_batch[0], 0, 3)).save(name, "PNG")

            step_size = 0.5 # the biggest change in the image will be this percent increase/decrease
            batch_updates = column.dream_batch(1)
            reshaped_batch += ((step_size*numpy.abs(reshaped_batch).max())/numpy.abs(batch_updates).max()) * batch_updates
            column.x_buffer.set_value(lasagne.utils.floatX(reshaped_batch), borrow=True)

            itr += 1

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

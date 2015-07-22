from os import listdir, path
import re
import numpy

import matplotlib
matplotlib.use('Agg')
from skimage.io import imread
matplotlib.rcParams.update({'font.size': 2})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import theano

import my_code.occluded_args as args
from my_code.predict import load_column, model_runid

import pdb

class OcclusionStudy(object):
    def __init__(self, data_stream, occlusion_patch_size, test_imagepath=None):
        """

        """
        assert(occlusion_patch_size % 2 == 1) # for simplicity
        self.ds = data_stream
        self.occlusion_patch_size = occlusion_patch_size
        self.test_imagepath = test_imagepath

        patched_img_pad = (occlusion_patch_size - 1) / 2
        patched_img_dim = data_stream.image_shape[0] - patched_img_pad*2
        self.num_patch_centers = (patched_img_dim)**2 # number of images in test set

        self.patch_starts = [divmod(n, patched_img_dim) for n in xrange(self.num_patch_centers)]

    def nth_patch(self, n):
        i_start,j_start = self.patch_starts[n]
        i_end = i_start + self.occlusion_patch_size
        j_end = j_start + self.occlusion_patch_size

        return(i_start,i_end,j_start,j_end)

    def buffer_occluded_dataset(self):
        assert(self.test_imagepath)
        img = self.ds.feed_image(image_name=self.test_imagepath, image_dir='')
        channel_means = img.mean(axis=(0,1))

        x_cache_block = numpy.zeros(((self.ds.cache_size,) + self.ds.image_shape), dtype=theano.config.floatX)
        n_full_cache_blocks, n_leftovers = divmod(self.num_patch_centers, self.ds.cache_size)
        for ith_cache_block in xrange(n_full_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.ds.cache_size
            idxs_to_full_dataset = list(range(ith_cache_block * self.ds.cache_size, ith_cache_block_end))
            for ci,n in enumerate(idxs_to_full_dataset):
                i_start,i_end,j_start,j_end = self.nth_patch(n)

                x_cache_block[ci, ...] = img
                x_cache_block[ci, i_start:i_end, j_start:j_end, :] = channel_means
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')
        # sneak the leftovers out, padded by the previous full cache block
        if n_leftovers:
            for ci, n in enumerate(list(xrange(ith_cache_block_end, len(self.patch_starts)))):
                i_start,i_end,j_start,j_end = self.nth_patch(n)

                x_cache_block[ci, ...] = img
                x_cache_block[ci, i_start:i_end, j_start:j_end, :] = channel_means
                idxs_to_full_dataset[ci] = n
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')

    def accumulate_patches_into_heatmaps(self, all_test_output, outpath_prefix=''):
        outpath = "plots/%s_%s.png" % (outpath_prefix, path.splitext(path.basename(self.test_imagepath))[0])
        # http://matplotlib.org/examples/axes_grid/demo_axes_grid.html
        fig = plt.figure()
        grid = AxesGrid(fig, 143, # similar to subplot(143)
                    nrows_ncols = (1, 1))
        orig_img = imread(self.test_imagepath+'.png')
        grid[0].imshow(orig_img)
        grid = AxesGrid(fig, 144, # similar to subplot(144)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.15,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )

        for klass in xrange(all_test_output.shape[1]):
            accumulator = numpy.zeros(self.ds.image_shape[:2])
            normalizer = numpy.zeros(self.ds.image_shape[:2])
            for n in xrange(self.num_patch_centers):
                i_start,i_end,j_start,j_end = self.nth_patch(n)

                accumulator[i_start:i_end, j_start:j_end] += all_test_output[n,klass]
                normalizer[i_start:i_end, j_start:j_end] += 1
            normalized_img = accumulator / normalizer
            im = grid[klass].imshow(normalized_img, interpolation="nearest", vmin=0, vmax=1)
            grid.cbar_axes[klass].colorbar(im)
        grid.axes_llc.set_xticks([])
        grid.axes_llc.set_yticks([])
        print("Saving figure as: %s" % outpath)
        plt.savefig(outpath, dpi=600, bbox_inches='tight')

def file_iter(test_path):
    """
    Iterates through full paths of images.
    """
    e = ValueError("'%s' is neither a file nor a directory of images" % test_path)
    if path.isdir(test_path):
        images = [path.splitext(f)[0] for f in listdir(test_path) if re.search('\.(jpeg|png)', f)]
        if not len(images):
            raise e
        for image in images:
            yield test_path + image
    elif path.isfile(test_path):
        yield path.splitext(test_path)[0]
    else:
        raise e

def plot_occluded_activations(model_file, test_path, patch_size, **kwargs):
    assert(model_file)
    runid = model_runid(model_file)

    column = load_column(model_file, **kwargs)

    os = OcclusionStudy(column.ds, patch_size)

    try:
        for path in file_iter(test_path):
            os.test_imagepath = path
            all_test_predictions, all_test_output = column.test(override_buffer=os.buffer_occluded_dataset, override_num_examples=os.num_patch_centers)
            os.accumulate_patches_into_heatmaps(all_test_output, runid)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Occlusion Study"
    print "Done"

if __name__ == '__main__':
    _ = args.get()

    plot_occluded_activations(model_file=_.model_file,
           test_path=_.test_path,
           patch_size=_.patch_size,
           train_dataset=_.train_dataset,
           center=_.center,
           normalize=_.normalize,
           train_flip=_.train_flip,
           test_dataset=None,
           random_seed=_.random_seed,
           valid_dataset_size=_.valid_dataset_size,
           filter_shape=_.filter_shape)

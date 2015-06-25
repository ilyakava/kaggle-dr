# -*- coding: utf-8 -*-
import sys
import csv
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from nyanbar import NyanBar
import uuid

from my_code.predict_util import binary_accuracy_precision

import pdb

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

INNER_RADIUS_FULL_SIZE_PHOTO = -2
OUTER_RADIUS_FULL_SIZE_PHOTO = 15

class FundusPhoto(object):
    """
    Our goal is to make false positives unlikely in this class
    """
    thresh_level = 5 # empirically, 10 is too high

    def __init__(self, img_path):
        """
        Notation: Thresholded image matrix T, is n x m. Comprised of pixels Tij.
        However when xy coordinates are involved, the origin is the center of the image.
        """
        self.I = imread(img_path)
        # TODO try Otsu
        self.T = self.I > FundusPhoto.thresh_level
        self.n,self.m,c = self.T.shape
        self.i_img_center = self.n // 2 - 1
        self.j_img_center = self.m // 2 - 1
        self.radius = self.estimate_radius()

    def plot(self, ijs=np.array((0,2)), title='', color='red', save_path=None):
        """
        :param ijs: n x 2 matrix of (i,j) coordinates into image
        """
        plt.imshow(self.I)
        plt.scatter(ijs[:,1], ijs[:,0], color=color, alpha=0.3)
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()

    def scan_perimeter_intensity(self, i_center, j_center, img_version, radius_additive=OUTER_RADIUS_FULL_SIZE_PHOTO, step_size=1):
        """
        :param radius_additive: a number between 10 and 25 works for the full sized originals

        :return: n x 4 matrix (degree, i, j, intensity) sorted by descending intensity
        """
        # degrees -> rads
        rads = [((np.pi*degree)/(180.0), degree) for degree in xrange(0,360,step_size)]
        # radius + rads -> x,y
        xys = [(pol2cart(self.radius + radius_additive, phi))+(degree,) for phi, degree in rads]
        # -> i,j
        ijs = [(y + i_center, x + j_center, degree) for x, y, degree in xys]
        # -> valid i,j
        valid_ijs = []
        for i, j, degree in ijs:
            if i > 0 and i < self.n and j > 0 and j < self.m:
                valid_ijs.append((int(i), int(j), degree))
        # -> intensity (on thresholded image)
        intensities = np.array([[degree, i, j, img_version[i,j,:].sum()] for i,j,degree in valid_ijs])
        # sort by intensity
        intensity_sort_idx = np.argsort(intensities[:,3])
        return intensities[intensity_sort_idx[::-1],:] # flip for descending

    def estimate_radius(self):
        """
        sets radius to half of greatest width
        """
        red = self.T[:,:,0] # empirically, the most reliable channel

        eye_radius = red.sum(axis=1).max() / 2
        return eye_radius

    def refine_center(self, search_size=100, step_size=5, radius_additive=INNER_RADIUS_FULL_SIZE_PHOTO):
        """
        Jitters a perimeter of fixed radius around the center of the image
        to come up with a new estimate for the eyeball center
        """
        iter = 0
        saves = np.zeros(((search_size*2)**2,3))
        for i_add in xrange(-search_size, search_size, step_size):
            for j_add in xrange(-search_size, search_size, step_size):
                perim_intensities = self.scan_perimeter_intensity(
                    self.i_img_center+i_add, self.j_img_center+j_add, self.T, radius_additive, step_size)
                saves[iter, 0] = self.i_img_center+i_add
                saves[iter, 1] = self.j_img_center+j_add
                saves[iter, 2] = perim_intensities[:,3].sum()
                iter += 1

        idxs = np.argsort(saves[:,2])
        winner_ij = saves[idxs[-1],:2]
        self.i_img_center = winner_ij[0]
        self.j_img_center = winner_ij[1]
        return(winner_ij[0], winner_ij[1])

    def has_notch(self, radius_additive=OUTER_RADIUS_FULL_SIZE_PHOTO):
        """
        Checks if there is an identification tab in the top right of the
        image. Tries to achieve few false positives.

        TODO: check for short substring of 1's in thresholded image
        to indicate peak, will help discard false positives
        """
        perim_info = self.scan_perimeter_intensity(self.i_img_center, self.j_img_center, self.I, radius_additive)
        top_perim_info = perim_info[:3,:]
        # re-sort top_perim_info by degrees
        top_perim_info = top_perim_info[np.argsort(top_perim_info[:,0]),:]
        top_degrees = top_perim_info[:,0]
        top_ij = np.array(top_perim_info[:,1:3], dtype=int)
        top_intensities = top_perim_info[:,3]

        # check that at least 2 points are adjacent
        diffs = np.diff(top_degrees)
        if not (diffs.min() == 1):
            return 0
        back_two = diffs.tolist().index(1)
        adjacents = slice(1,3) if back_two else slice(0,2)
        # check that at least 2 top points are on the right side of img
        shifted_top_degrees = (top_degrees + 90) % 360
        if not (sum(shifted_top_degrees < 180) >= 2):
            return 0
        # check that level of adjacents is high enough
        adjacent_ijs = top_ij[adjacents,:]
        adjacent_mean_intensities = self.I[adjacent_ijs.T[0], adjacent_ijs.T[1],:].mean(axis=1)
        if not sum(adjacent_mean_intensities > 15) > 0:
            return 0
        return 1

class Ophthalmoscope(object):
    """
    Can be indirect (produce inverted images) by providing an identification
    tab in the top right of the image, or direct. Each ophthalmoscope takes
    at least one photo (should be two though) that it produced.
    """
    def __init__(self, img_paths):
        self.photos = [FundusPhoto(photo) for photo in img_paths]
        for photo in self.photos:
            photo.refine_center()

    def direct(self):
        """
        If image has a notch then it is not inverted (xxxx_right.jpeg means
        optic nerve is on the right) and was produced by a direct
        ophthalmoscope.
        """
        n_notches = sum([photo.has_notch() for photo in self.photos])
        if n_notches > 0:
            return 1
        else:
            return 0

class FundusPhotoAligner(object):
    """
    Outputs a csv with details on what combination of flips to perform per
    image (horizontal, vertical) to place the optic nerve on the right side.
    """
    def __init__(self, img_paths, outfile_name=("data/align/fundus_photo_aligner_%s.csv" % str(uuid.uuid4())[:8])):
        path = '/'.join(img_paths[0].split('/')[:-1])
        img_names = [img_path.split('/')[-1] for img_path in img_paths]
        ext = img_names[0].split('.')[-1]
        ids = set([img_path.split('_')[0] for img_path in img_names])
        assert((len(ids) * 2) == len(img_names))
        grouped_img_names = [["%s_left.%s" % (id,ext), "%s_right.%s" % (id,ext)] for id in ids]
        self.labels = {}

        print "| " + ("⚐ ⚑ " * 19) + "|"
        pb = NyanBar(tasks=len(grouped_img_names))
        with open(outfile_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image','horizontal_flip','vertical_flip'])
            for img_names in grouped_img_names:
                img_paths = ["%s/%s" % (path, name) for name in img_names]
                fundus_photos = [FundusPhoto(img_path) for img_path in img_paths]
                for i, fundus_photo in enumerate(fundus_photos):
                    fundus_photo.refine_center()
                    direct = fundus_photo.has_notch()
                    lens_flip_vector = np.array([0,0] if direct else [1,1])
                    left_right_flip_vector = np.array([0,1] if i == 0 else [0,0])
                    final_flip_vector = (lens_flip_vector + left_right_flip_vector) % 2
                    writerow = [img_names[i]] + final_flip_vector.tolist()
                    writer.writerow(writerow)
                    self.labels[img_names[i]] = direct
                pb.task_done()
        pb.finish()
        print("Done! created: %s" % outfile_name)

def predict_single_image(name, title_prefix='', save_path=None, true_label=None):
    fp = FundusPhoto(name)
    winner_ij = fp.refine_center()

    perim_info = fp.scan_perimeter_intensity(winner_ij[0], winner_ij[1], fp.I)
    top_perim_info = perim_info[:3,:] # limit here
    predicted_label = fp.has_notch()
    if type(true_label) is int:
        extra_prefix = "CORRECT" if true_label == predicted_label else "INCORRECT"
    else:
        extra_prefix = "Predicted %i" % predicted_label
    title = "%s. top degrees: {} of avg brightness %.1f. %s".format(top_perim_info[:,0]) % (extra_prefix, top_perim_info[:,3].mean(), title_prefix)
    top_ij = perim_info[:,1:3]
    fp.plot(top_ij, title=title, color='green', save_path=save_path)
    return predicted_label

if __name__ == '__main__':
    # Three running modes for testing the identification tab detector

    if len(sys.argv) == 2: # single image
        predict_single_image(sys.argv[1])

    elif len(sys.argv) == 4: # batch of single images
        label_csv = sys.argv[1]
        inpath = sys.argv[2]
        outpath = sys.argv[3]

        true_labels = []
        predicted_labels = []

        with open(label_csv, 'rU') as csvfile:
            reader = csv.reader(csvfile, dialect=csv.excel_tab, delimiter=',')
            next(reader, None) # skip header
            print "| " + ("⚐ ⚑ " * 19) + "|"
            pb = NyanBar(tasks=(sum(1 for row in open(label_csv, 'rU'))-1))
            for row in reader:
                imgname = row[0]
                true_label = 1 if (row[1] == 'tr' or row[1] == 'br' or row[1] == '1') else 0

                title_prefix = "True Label: %s" % true_label
                predicted_label = predict_single_image(inpath + imgname, title_prefix=title_prefix, save_path=(outpath + imgname), true_label=true_label)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

                pb.task_done()
            pb.finish()
            accuracy, precision = binary_accuracy_precision(true_labels, predicted_labels)
            print("Accuracy: %.3f. Precision: %.3f." % (accuracy, precision))
    elif len(sys.argv) == 3: # batch of doubled images
        true_label_csv = sys.argv[1]
        inpath = sys.argv[2]

        true_labels = []
        img_names = [] # for reference of label order
        img_paths = []
        # read csv
        # we're going to assume true_label_csv has the double images next to each other
        # whether left/right comes first doesn't matter since their label is identical
        with open(true_label_csv, 'rU') as csvfile:
            reader = csv.reader(csvfile, dialect=csv.excel_tab, delimiter=',')
            next(reader, None) # skip header
            for row in reader:
                img_name = row[0]
                true_label = int(row[1])

                true_labels.append(true_label)
                img_names.append(img_name)
                img_paths.append("%s%s" % (inpath, img_name))
        # run
        fpa = FundusPhotoAligner(img_paths)
        # compare
        predicted_labels = [fpa.labels[img_name] for img_name in img_names]
        accuracy, precision = binary_accuracy_precision(true_labels, predicted_labels)
        print("Accuracy: %.3f. Precision: %.3f." % (accuracy, precision))
    else:
        ValueError("%i argument number not supported" % len(sys.argv))


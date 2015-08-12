import unittest
import numpy

from my_code.plot_dreams import calculate_octave_and_tile_sizes

class CalculateOctaveAndTileSizes(unittest.TestCase):
    def tile_within_octave_image(self, octave_sizes, octave_tile_corners, source_size, nn_image_size):
        for i, octave_tiles in enumerate(octave_tile_corners):
            max_h, max_w = octave_sizes[i]
            for tile in octave_tiles:
                t,l = tile
                self.failUnless(t + nn_image_size <= max_h)
                self.failUnless(l + nn_image_size <= max_w)

    def full_octave_image_coverage(self, octave_sizes, octave_tile_corners, source_size, nn_image_size):
        for i, octave_tiles in enumerate(octave_tile_corners):
            octave_image = numpy.zeros(octave_sizes[i])
            for tile in octave_tiles:
                t,l = tile
                b,r = [d+nn_image_size for d in tile]
                octave_image[t:b, l:r] += numpy.ones((nn_image_size,nn_image_size))
            self.failUnless((octave_image == 0).sum() == 0)

    def _test(self,source_size,nn_image_size):
        octave_sizes, octave_tile_corners = calculate_octave_and_tile_sizes(source_size, nn_image_size)
        self.failUnless(octave_sizes[0] == list(source_size))
        self.tile_within_octave_image(octave_sizes, octave_tile_corners, source_size, nn_image_size)
        self.full_octave_image_coverage(octave_sizes, octave_tile_corners, source_size, nn_image_size)

    def testSimpleSquare(self):
        self._test(source_size=(11,11), nn_image_size=4)

    def testSimple(self):
        self._test(source_size=(10,11), nn_image_size=4)

    def testRealMedium(self):
        self._test(source_size=(252,260), nn_image_size=192)

    def testRealTall(self):
        self._test(source_size=(400,264), nn_image_size=192)

    def testRealWide(self):
        self._test(source_size=(250,640), nn_image_size=192)

def main():
  unittest.main()

if __name__ == '__main__':
  main()

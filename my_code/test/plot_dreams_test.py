import unittest
import numpy

from my_code.plot_dreams import calculate_octave_and_tile_sizes

class CalculateOctaveAndTileSizes(unittest.TestCase):
    def tile_within_image(self, octave_tile_corners, source_size, nn_image_size):
        max_h, max_w = source_size
        for octave_tiles in octave_tile_corners:
            for tile in octave_tiles:
                t,l = tile
                self.failUnless(t + nn_image_size <= max_h)
                self.failUnless(l + nn_image_size <= max_w)

    def full_image_coverage(self, octave_tile_corners, source_size, nn_image_size):
        source = numpy.zeros(source_size)
        max_h, max_w = source_size
        for octave_tiles in octave_tile_corners:
            for tile in octave_tiles:
                t,l = tile
                b,r = [d+nn_image_size for d in tile]
                source[t:b, l:r] += numpy.ones((nn_image_size,nn_image_size))
        self.failUnless((source == 0).sum() == 0)

    def _test(self,source_size,nn_image_size):
        octave_sizes, octave_tile_corners = calculate_octave_and_tile_sizes(source_size, nn_image_size)
        self.tile_within_image(octave_tile_corners, source_size, nn_image_size)
        self.full_image_coverage(octave_tile_corners, source_size, nn_image_size)

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

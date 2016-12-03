import unittest

import numpy as np
import tensorflow as tf

from itertools import izip

from ..networks import rpn

class generateAnchorsTest(unittest.TestCase):

	def test_properNumberOfAnchors(self):
		output = rpn.generateAnchors()
		self.assertEqual(9,len(output))

	def test_generateAnchors(self):
		output = rpn.generateAnchors(ratios=[2,1,.5])
		expected = [
			[ -83.0,  -39.0,  100.0,   56.0],
			[-175.0,  -87.0,  192.0,  104.0],
			[-359.0, -183.0,  376.0,  200.0],
			[ -55.0,  -55.0,   72.0,   72.0],
			[-119.0, -119.0,  136.0,  136.0],
			[-247.0, -247.0,  264.0,  264.0],
			[ -35.0,  -79.0,   52.0,   96.0],
			[ -79.0, -167.0,   96.0,  184.0],
			[-167.0, -343.0,  184.0,  360.0]]

		for rowOut, rowExp in izip(output,expected):
			for eleOut, eleExp in izip(rowOut,rowExp):
				self.assertAlmostEqual(eleOut,eleExp, places=5)

if __name__=="__main__":
	unittest.main()

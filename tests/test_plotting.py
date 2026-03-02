# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import os
import unittest
import tempfile
import shutil
from vlbench.plotting.calibration import bins2diagram


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_bins2diagram_generates_pdf(self):
        # Sample bins data: (bincounts, corrects, cumconf)
        # 10 bins, 10 samples each, mixed results
        bins = ([10] * 10, [i for i in range(10)], [float(i) for i in range(10)])

        pdf_path = os.path.join(self.tmpdir, "test_diagram.pdf")
        bins2diagram(bins, displays=False, saveas=pdf_path)

        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 0)


if __name__ == "__main__":
    unittest.main()

import os
import pathlib
import shutil
import unittest

import kdd98.data_loader as dl
from kdd98.data_loader import Cleaner
from kdd98.config import Config


class TestCleanerSetup(unittest.TestCase):

    path = pathlib.Path(os.path.realpath(__file__), "/test_clean")

    def setUp(self):
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        #shutil.copyfile(pathlib.Path(pathlib.Path(Config.get("data_dir").resolve()), "cup98LRN_snip.txt"), pathlib.Path(self.path, "cup98LRN_snip.txt"))
        #self.old_data_dir = Config.get("data_dir")
        #Config.set("data_dir", os.path.dirname(os.path.realpath(__file__))+"/test_clean")

    def tearDown(self):
        shutil.rmtree(self.path)
        #Config.set("data_dir", self.old_data_dir)

class TestCleaner(TestCleanerSetup):

    def test_clean(self):
        data_loader = dl.KDD98DataLoader("cup98LRN_snip.txt")
        data = data_loader.clean_data
        print(data.info())

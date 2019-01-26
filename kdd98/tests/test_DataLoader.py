import os
import pathlib
import shutil
import unittest

import kdd98.data_loader as dl
from kdd98.config import App


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(os.path.realpath(__file__))+"/test_download"
        if not os.path.isdir(path):
            os.mkdir(path)

    def tearDown(self):
        path = os.path.dirname(os.path.realpath(__file__))+"/test_download"
        shutil.rmtree(path)

class TestDataLoaderDownload(TestDataLoader):

    def test_retrieve_online(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        loader = dl.KDD98DataLoader(App.config("learn_file_name"))
        loader.fetch_online(dl_dir="./test_download")

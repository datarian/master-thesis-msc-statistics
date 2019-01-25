import unittest
import data_loader
import os
import pathlib


class TestDataLoader(unittest.TestCase):

    def test_retrieve_online(self):
        os.chdir(__file__+"/test_download")
        dl = KDD98DataLoader()
        dl.fetch_online()

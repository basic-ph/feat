# -*- coding: utf-8 -*-

import json
import sys

import meshio
import pytest

sys.path.insert(0, "/home/basic-ph/thesis/feat/feat/")


@pytest.fixture
def setup_mesh():
    def _load_mesh(path):
        mesh = meshio.read(path)
        return mesh

    return _load_mesh


@pytest.fixture
def setup_data():
    def _load_data(path):
        with open(path, "r") as data_file:
            data = json.load(data_file)
        return data

    return _load_data

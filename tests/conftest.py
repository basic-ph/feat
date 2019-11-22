# -*- coding: utf-8 -*-

import json
import os
import sys

import meshio
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "feat"))

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

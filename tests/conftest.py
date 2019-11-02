# -*- coding: utf-8 -*-

import json

import meshio
import pytest


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

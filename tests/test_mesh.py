import pytest

from feat import mesh


def test_center_in_box():
    vertex = [0.0, 0.0, 0.0]
    side = 5.0
    x = 3.0; y = 5.5
    assert not mesh.center_in_box(x, y, vertex, side)


def test_circle_intersect_side():
    x = 12.100693005155364
    y = 15.265433923836474
    radius = 1.0
    x1 = 15.25; y1 = 15.25
    x2 = 7.62 ; y2 = 15.25
    assert mesh.circle_insersect_side(x, y, radius, x1, y1, x2, y2)
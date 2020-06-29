import pytest

from feat import mesh


def test_center_in_box():
    vertex = [0.0, 0.0, 0.0]
    side = 5.0
    x1 = 3.0; y1 = 5.5
    x2 = 2.5; y2 = 3.0
    assert not mesh.center_in_box(x1, y1, vertex, side)
    assert mesh.center_in_box(x2, y2, vertex, side)


def test_circle_intersect_side():
    x = 12.100693005155364
    y = 15.265433923836474
    radius = 1.0
    x1 = 15.25; y1 = 15.25
    x2 = 7.62 ; y2 = 15.25
    assert mesh.circle_insersect_side(x, y, radius, x1, y1, x2, y2)
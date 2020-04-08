import sys
import math
from pathlib import Path

import meshio
import numpy as np
import pygmsh


def get_fiber_centers(radius, number, side, min_distance, offset, max_iter):
   
    rg = np.random.default_rng()  # random generator, accept seed as arg (reproducibility)
    get_dist = lambda x_0, y_0, x_1, y_1: np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

    x_array = np.zeros(number)  # array with x coordinate of centers
    y_array = np.zeros(number)  # array with y coordinate of centers
    i = 0  # counter for array indexing
    k = 0  # iterations counter

    while k < max_iter:
        k += 1
        valid = True
        x = offset + (side - 2*offset)* rg.random()
        y = offset + (side - 2*offset)* rg.random()

        for j in range(i):
            distance = get_dist(x, y, x_array[j], y_array[j])
            if distance > min_distance:
                valid = True
            else:
                valid = False
                break  # exit the loop when the first intersection is found

        if valid:  # if no intersection is found center coordinates are added to arrays
            x_array[i] = x
            y_array[i] = y
            i += 1
        
        if i == number:
            break

    if i < (number -1):
        sys.exit()

    return x_array, y_array


def create_mesh(geo_path, msh_path, radius, number, side, x_array, y_array, coarse_cl, fine_cl):
    
    geom = pygmsh.built_in.Geometry()

    # RVE square geometry
    p0 = geom.add_point([0.0, 0.0, 0.0], coarse_cl)
    p1 = geom.add_point([side, 0.0, 0.0], coarse_cl)
    p2 = geom.add_point([side, side, 0.0], coarse_cl)
    p3 = geom.add_point([0.0, side, 0.0], coarse_cl)
    
    l0 = geom.add_line(p0, p1)
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p0)
    square_loop = geom.add_line_loop([l0, l1, l2, l3])

    # Fibers geometry
    circle_arcs = []  # list of circle arcs that form fibers
    circle_loops = []  # list of gmsh line loops for fibers
    fiber_surfaces = []  # list of gmsh surfaces related to fibers

    for n in range(number):
        center = geom.add_point([x_array[n], y_array[n], 0], coarse_cl)
        circle = geom.add_circle([x_array[n], y_array[n], 0], radius, lcar=coarse_cl)
        circle_loops.append(circle.line_loop)
        geom.add_raw_code(f"Point{{{center.id}}} In Surface{{{circle.plane_surface.id}}};")
        fiber_surfaces.append(circle.plane_surface)
        arcs = [line.id for line in circle.line_loop.lines]
        circle_arcs.extend(arcs)

    square_surface = geom.add_plane_surface(square_loop, holes=circle_loops)  # create matrix surface subtracting circles from square geometry

    # Mesh size Fields (http://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes)
    geom.add_raw_code("Field[1] = Distance;")
    geom.add_raw_code("Field[1].NNodesByEdge = 100;")
    geom.add_raw_code("Field[1].EdgesList = {{{}}};".format(", ".join(circle_arcs)))
    geom.add_raw_code("Field[2] = Threshold;\n"
        "Field[2].IField = 1;\n"
        f"Field[2].LcMin = {fine_cl};\n"
        f"Field[2].LcMax = {coarse_cl};\n"
        "Field[2].DistMin = 0.01;\n"
        f"Field[2].DistMax = {radius};\n"  # FIXME test this
        "Background Field = 2;\n"
    )
    geom.add_raw_code("Mesh.CharacteristicLengthExtendFromBoundary = 0;\n"
        "Mesh.CharacteristicLengthFromPoints = 0;\n"
        "Mesh.CharacteristicLengthFromCurvature = 0;\n"
    )

    # Physical groups
    geom.add_physical(square_surface, label="matrix")
    geom.add_physical(fiber_surfaces, label="fiber")
    geom.add_physical(l3, label="left side")  # constrained left side
    geom.add_physical(p0, label="bottom left corner")  # fixed corner
    geom.add_physical(l1, label="right side")  # imposed displacement right side

    # Gmsh .msh file generation
    mesh = pygmsh.generate_mesh(
        geom,
        # geo_filename=str(geo_path),  # uncomment this for saving geo and msh
        # msh_filename=str(msh_path),
        verbose=False,
        dim=2,
    )
    return mesh  # returning meshio Mesh object for further needs


if __name__ == "__main__":

    geo_path = "data/geo/rve_bench.geo"
    msh_path = "data/msh/rve_bench.msh"
    max_iter = 100000

    # RVE logic
    Vf = 0.30  # fiber volume fraction
    radius = 1.0
    number = 50
    side = math.sqrt(math.pi * radius**2 * number / Vf)
    print("rve side = ", side)
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    coarse_cl = 0.5
    fine_cl = coarse_cl / 5

    x_array, y_array = get_fiber_centers(radius, number, side, min_distance, offset, max_iter)

    mesh = create_mesh(
        geo_path,
        msh_path,
        radius,
        number,
        side,
        x_array,
        y_array,
        coarse_cl,
        fine_cl
    )

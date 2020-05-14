import sys
import math
from pathlib import Path
import logging

import meshio
import numpy as np
import pygmsh


logger = logging.getLogger(__name__)


def get_fiber_centers(rand_gen, radius, number, side, min_distance, offset, max_iter, x_list, y_list, old_side):
   
    get_dist = lambda x_0, y_0, x_1, y_1: math.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

    i = len(x_list)  # counter for array indexing
    k = 0  # iterations counter

    while k < max_iter:
        k += 1
        valid = True
        x = offset + (side - 2*offset)* rand_gen.random()
        y = offset + (side - 2*offset)* rand_gen.random()

        # logger.debug("oldside: %s", old_side)
        # logger.debug("x y: %s %s", x, y)
        # check center outside old domain
        if old_side is not None:
            if (x < old_side) and (y < old_side):
                # logger.debug("skip current (x,y)")
                continue
        # logger.debug("checking superposition...")

        # check superposition with other fibers
        for j in range(i):
            distance = get_dist(x, y, x_list[j], y_list[j])
            if distance > min_distance:
                valid = True
            else:
                valid = False
                break  # exit the loop when the first intersection is found

        if valid:  # if no intersection is found center coordinates are added to arrays
            x_list.append(x)
            y_list.append(y)
            i += 1
        
        if i == number:
            break

    if i < (number):
        logger.warning("Fiber centers not found!!! exit...")
        sys.exit()

    return x_list, y_list


def create_mesh(geo_path, msh_path, radius, number, side, x_list, y_list, coarse_cl, fine_cl):
    
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
        center = geom.add_point([x_list[n], y_list[n], 0], coarse_cl)
        circle = geom.add_circle([x_list[n], y_list[n], 0], radius, lcar=coarse_cl)
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
        msh_filename=str(msh_path),
        verbose=False,
        dim=2,
    )
    return mesh  # returning meshio Mesh object for further needs


if __name__ == "__main__":
    # logger
    log_lvl = logging.DEBUG
    root_logger = logging.getLogger()
    root_logger.setLevel(log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    geo_path = "../data/geo/refined_2.geo"
    msh_path = "../data/reboot/validation.msh"
    max_iter = 100000

    # RVE logic
    Vf = 0.30  # fiber volume fraction
    radius = 1.0
    number = 10
    side = math.sqrt(math.pi * radius**2 * number / Vf)
    print("rve side = ", side)
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    coarse_cl = 0.5
    fine_cl = coarse_cl / 2

    rg = np.random.default_rng(19)  # random generator, accept seed as arg (reproducibility)

    x_list = [
        4.477015140636965, 3.300070757836716, 7.373705098213486, 4.391170095684478,
        5.465886648631734, 2.446386464460651, 1.2239329865870123, 8.586118632630855,
        8.973020999749487, 1.4037057884961852
    ]
    y_list = [
        8.537755727601748, 1.582386474665313, 5.427519658891324, 6.026362684282338,
        1.9917791185075826, 7.664701411635054, 5.76315679939553, 2.851615342928181,
        9.055487089785053, 2.581815590004845
    ]
    x_list, y_list = get_fiber_centers(rg, radius, number, side, min_distance, offset, max_iter, x_list, y_list)
    root_logger.debug(x_list)
    root_logger.debug(y_list)
    mesh = create_mesh(
        geo_path,
        msh_path,
        radius,
        number,
        side,
        x_list,
        y_list,
        coarse_cl,
        fine_cl
    )

import sys
import math
import logging

import numpy as np
import pygmsh

logger = logging.getLogger(__name__)


def center_in_box(x, y, vertex, side):
    return (
        vertex[0] < x < vertex[0] + side
    ) and (
        vertex[1] < y < vertex[1] + side
    )


def circle_insersect_side(x, y, radius, x1, y1, x2, y2):
    """Weisstein, Eric W. "Circle-Line Intersection." 
    From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    X1 = x1 - x; Y1 = y1 - y
    X2 = x2 - x; Y2 = y2 - y

    dx = X2-X1; dy = Y2-Y1
    dr = math.sqrt(dx**2 + dy**2)
    D = X1*Y2 - X2*Y1
    delta = radius**2 * dr**2 - D**2

    if delta > 0:
        # first intersection point
        Xa = (D*dy + math.copysign(dx,dy)*math.sqrt(delta)) / (dr**2)
        Ya = (-D*dx + abs(dy)*math.sqrt(delta)) / (dr**2)
        # second intersection point
        Xb = (D*dy - math.copysign(dx,dy)*math.sqrt(delta)) / (dr**2)
        Yb = (-D*dx - abs(dy)*math.sqrt(delta)) / (dr**2)

        Xa_collide = (min(X1,X2) <= Xa <= max(X1,X2))
        Ya_collide = (min(Y1,Y2) <= Ya <= max(Y1,Y2))
        
        Xb_collide = (min(X1,X2) <= Xb <= max(X1,X2))
        Yb_collide = (min(Y1,Y2) <= Yb <= max(Y1,Y2))

        return (Xa_collide and Ya_collide) or (Xb_collide and Yb_collide)

    else:  # delta <= 0 are considered not colliding
        return False


def circle_intersect_box(x, y, radius, vertex, side):
    # 1st check center of circle is inside the box?
    check1 = center_in_box(x, y, vertex, side)
    logger.debug("check1: %s", check1)

    if check1:
        return True

    # 2nd check: the circle and each side of the box have intersection?
    x1 = vertex[0]; y1 = vertex[1]; x2 = vertex[0]+side; y2 = vertex[1]
    side1 = circle_insersect_side(x, y, radius, x1, y1, x2, y2)
    logger.debug("side1 intersect: %s", side1)
    
    x1 = vertex[0]+side; y1 = vertex[1]; x2 = vertex[0]+side; y2 = vertex[1]+side
    side2 = circle_insersect_side(x, y, radius, x1, y1, x2, y2)
    logger.debug("side2 intersect: %s", side2)
    
    x1 = vertex[0]+side; y1 = vertex[1]+side; x2 = vertex[0]; y2 = vertex[1]+side
    side3 = circle_insersect_side(x, y, radius, x1, y1, x2, y2)
    logger.debug("side3 intersect: %s", side3)
    
    x1 = vertex[0]; y1 = vertex[1]+side; x2 = vertex[0]; y2 = vertex[1]
    side4 = circle_insersect_side(x, y, radius, x1, y1, x2, y2)
    logger.debug("side4 intersect: %s", side4)

    check2 = (side1 or side2 or side3 or side4)  # circle intersect one of ths sides?
    logger.debug("check2: %s", check2)

    return (check1 or check2)



def get_fiber_centers(rand_gen, radius, number, max_side, vertex, actual_side, min_distance, offset, max_iter, centers):
   
    get_dist = lambda x_0, y_0, x_1, y_1: math.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

    i = 0  # = 0 | counter for array indexing
    k = 0  # iterations counter

    while k < max_iter:
        k += 1
        valid = True
        x = offset + (max_side - 2*offset)* rand_gen.random()
        y = offset + (max_side - 2*offset)* rand_gen.random()

        # check superposition with other fibers
        if centers:  # skip if center = []
            for j in range(len(centers)):
                distance = get_dist(x, y, centers[j][0], centers[j][1])
                if distance > min_distance:
                    valid = True
                else:
                    valid = False
                    break  # exit the loop when the first intersection is found

        logger.debug("x: %s, y: %s", x, y)
        included = circle_intersect_box(x, y, radius, vertex, actual_side)
        logger.debug("included: %s", included)
        logger.debug("")

        if valid:  # if no intersection is found center coordinates are added to arrays
            i += 1
            if included:  # has intersection with the actual box
                centers.append([x, y, 0.0])

        if i == number:
            break

    if i < (number):
        logger.warning("Fiber centers not found!!! exit...")
        sys.exit()

    return centers


def create_mesh(geo_path, msh_path, radius, number, vertex, side, centers, coarse_cl, fine_cl):

    geom = pygmsh.opencascade.Geometry()

    disks = []
    for i in range(len(centers)):
        fiber = geom.add_disk(centers[i], radius)
        disks.append(fiber)
    
    disk_tags = ", ".join([d.id for d in disks])

    rectangle = geom.add_rectangle(vertex, side, side)

    geom.add_raw_code(
        f"BooleanIntersection{{ Surface{{{disk_tags}}}; Delete; }} "
        f"{{ Surface{{{rectangle.id}}}; }}"
    )
    geom.add_raw_code(
        f"BooleanDifference{{ Surface{{{rectangle.id}}}; Delete; }} "
        f"{{ Surface{{{disk_tags}}}; }}"
    )

    e = 0.01
    geom.add_raw_code(
        f"p() = Point In BoundingBox"
        f"{{{vertex[0]-e}, {vertex[1]-e}, {vertex[2]-e}, {vertex[0]+e}, {vertex[1]+e}, {vertex[2]+e}}};"  # bottom left corner
    )
    geom.add_raw_code(
        f"q() = Curve In BoundingBox"
        f"{{{vertex[0]-e}, {vertex[1]-e}, {vertex[2]-e}, {vertex[0]+e}, {side+e}, {vertex[2]+e}}};"  # left side
    )
    geom.add_raw_code(
        f"r() = Curve In BoundingBox"
        f"{{{side-e}, {vertex[1]-e}, {vertex[2]-e}, {side+e}, {side+e}, {vertex[2]+e}}};"  # right side
    )

    geom.add_raw_code(
        f"boundary[] = Boundary{{ Surface{{{disk_tags}}}; }};"  # identify boundaries of fibers for mesh refinement
    )

    geom.add_raw_code("Field[1] = Distance;")
    geom.add_raw_code("Field[1].NNodesByEdge = 100;")
    geom.add_raw_code(f"Field[1].EdgesList = {{boundary[]}};")
    geom.add_raw_code(
        "Field[2] = Threshold;\n"
        "Field[2].IField = 1;\n"
        f"Field[2].LcMin = {fine_cl};\n"
        f"Field[2].LcMax = {coarse_cl};\n"
        "Field[2].DistMin = 0.01;\n"
        f"Field[2].DistMax = {radius};\n"
        "Background Field = 2;"
    )
    geom.add_raw_code(
        "Mesh.CharacteristicLengthExtendFromBoundary = 0;\n"
        "Mesh.CharacteristicLengthFromPoints = 0;\n"
        "Mesh.CharacteristicLengthFromCurvature = 0;"
    )

    geom.add_raw_code(
        f"Physical Surface(\"matrix\") = {{{rectangle.id}}};\n"
        f"Physical Surface(\"fiber\") = {{{disk_tags}}};\n"
        f"Physical Point(\"bottom left corner\") = {{p()}};\n"
        f"Physical Line(\"left side\") = {{q()}};\n"
        f"Physical Line(\"right side\") = {{r()}};\n"
    )
    
    mesh = pygmsh.generate_mesh(
        geom,
        geo_filename=str(geo_path),  # uncomment this for saving geo and msh
        msh_filename=str(msh_path),
        verbose=False,
        dim=2,
    )
    return mesh


if __name__ == "__main__":
    # logger
    log_lvl = logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    geo_path = "../data/geo/auto_cake.geo"
    msh_path = "../data/msh/auto_cake.msh"
    max_iter = 1000000

    # RVE logic
    Vf = 0.45  # fiber volume fraction
    radius = 1.0
    number = 150
    side = math.sqrt(math.pi * radius**2 * number / Vf)
    root_logger.info("rve side = %s", side)
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    coarse_cl = 0.5
    fine_cl = coarse_cl / 5

    rg = np.random.default_rng(96)  # random generator, accept seed as arg (reproducibility)

    
    s = 0
    step = side / (10 *2)
    root_logger.info("step: %s", step)
    vertex = [step*s, step*s, 0.0]
    # vertex = [2.5, 2.5, 0.0]
    actual_side = side - s*(2*step)
    root_logger.info("actual side: %s", actual_side)
    centers = []
    # centers = [
    #     [0.0, 0.0, 0.0],
    #     [5.0, 5.0, 0.0],
    #     [0.0, 5.0, 0.0],
    #     [10.0, 5.0, 0.0],
    # ]
    centers = get_fiber_centers(rg, radius, number, side, vertex, actual_side, min_distance, offset, max_iter, centers)
    root_logger.debug(centers)
    root_logger.info(len(centers))
    mesh = create_mesh(
        geo_path,
        msh_path,
        radius,
        number,
        vertex,
        actual_side,
        centers,
        coarse_cl,
        fine_cl
    )
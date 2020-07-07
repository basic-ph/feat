import logging
import math
import time

import numpy as np

from feat import mesh

logger = logging.getLogger(__name__)


get_dist = lambda x_0, y_0, x_1, y_1: math.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

get_dist_squared = lambda x_0, y_0, x_1, y_1: (x_0 - x_1)**2 + (y_0 - y_1)**2


def get_positions(rand_gen, number, radius, vertex, side, min_dist, max_iter):

    centers = []
    i = 0  # valid positions counter
    k = 0  # iteration counter
    while (k < max_iter) and (i < number):
        k += 1

        x = -radius + (side+radius) * rand_gen.random()  # could be added box corner coordinate
        y = -radius + (side+radius) * rand_gen.random()

        if mesh.circle_intersect_box(x, y, radius, vertex, side):
            i += 1
            centers.append([x, y, 0.0])


    if i < (number):
        logger.warning("Fiber centers not found!!! exit...")
        sys.exit()

    return centers


def get_circle_side_intersection(x, y, radius, x1, y1, x2, y2):
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

        tol = 1e-9
        Xa_collide = (min(X1,X2)-tol <= Xa <= max(X1,X2)+tol)
        Ya_collide = (min(Y1,Y2)-tol <= Ya <= max(Y1,Y2)+tol)
        
        Xb_collide = (min(X1,X2)-tol <= Xb <= max(X1,X2)+tol)
        Yb_collide = (min(Y1,Y2)-tol <= Yb <= max(Y1,Y2)+tol)

        if (Xa_collide and Ya_collide):
            if (Xb_collide and Yb_collide):
                return Xa, Ya, Xb, Yb
            else:
                return Xa, Ya
        elif (Xb_collide and Yb_collide):
            return Xb, Yb
    else:
        return None


def get_included_area(x, y, radius, vertex, side):
    """Compute only the area of a circle included inside a square
    Weisstein, Eric W. "Circular Segment." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/CircularSegment.html
    """
    sides = [ # [x1, y1, x2, y2]
        [vertex[0], vertex[1], vertex[0]+side, vertex[1]],
        [vertex[0]+side, vertex[1], vertex[0]+side, vertex[1]+side],
        [vertex[0]+side, vertex[1]+side, vertex[0], vertex[1]+side],
        [vertex[0], vertex[1]+side, vertex[0], vertex[1]],
    ]
    corners = [
        [vertex[0], vertex[1]],
        [vertex[0]+side, vertex[1]],
        [vertex[0]+side, vertex[1]+side],
        [vertex[0], vertex[1]+side]
    ]
    tot_intersect = []
    # print(f"x: {x}, y: {y}")
    for s in sides:
        # print(f"side: ({s[0]}, {s[1]}, {s[2]}, {s[3]})")
        intersect = get_circle_side_intersection(
            x, y, radius, s[0], s[1], s[2], s[3]
        )
        # print(f"intersect {intersect}")
        if intersect is not None:
            tot_intersect.append(intersect)
    # print(f"tot intersect: {tot_intersect}")
    if len(tot_intersect) == 1:  # both intersections on the same side
        Xa = tot_intersect[0][0]; Ya = tot_intersect[0][1]
        Xb = tot_intersect[0][2]; Yb = tot_intersect[0][3]
        
        if Xa == Xb:
            r = Xa  # this should be |x - xa| but we are in the coordinate system of the center (x,y) so x=0
            h = radius - r  # TODO remove prev line (useless)
            segment_area = radius**2 * math.acos((radius-h)/radius) - (radius-h) * math.sqrt(2*radius*h - h**2)
            if mesh.center_in_box(x, y, vertex, side):
                included_area = math.pi * radius**2 - segment_area
            else:
                included_area = segment_area
        elif Ya == Yb:
            r = Ya  # this should be |x - xa| but we are in the coordinate system of the center (x,y) so x=0
            h = radius - r  # TODO remove prev line (useless)
            segment_area = radius**2 * math.acos((radius-h)/radius) - (radius-h) * math.sqrt(2*radius*h - h**2)
            if mesh.center_in_box(x, y, vertex, side):
                included_area = math.pi * radius**2 - segment_area
            else:
                included_area = segment_area
    elif len(tot_intersect) == 2:  # intersections on different sides (corner collision)
        Xa = tot_intersect[0][0]; Ya = tot_intersect[0][1]
        Xb = tot_intersect[1][0]; Yb = tot_intersect[1][1]

        corner_dist = [get_dist(x,y,c[0],c[1]) for c in corners]
        # print(corner_dist)
        corner_index = corner_dist.index(min(corner_dist))
        # print(corner_index)
        Xc = corners[corner_index][0] - x  # converted to the coordinate system of the center (x,y)
        Yc = corners[corner_index][1] - y

        if Xc == Xa:
            ac_side = abs(Yc - Ya)
            bc_side = abs(Xb - Xc)
        else:  # Xc == Xb
            ac_side = abs(Xc - Xa)
            bc_side = abs(Yb - Yc)
        triangle_area = 0.5 * ac_side * bc_side  # triangle is surely rectangle with AC and BC catheti

        Xm = (Xa + Xb)/2; Ym = (Ya + Yb)/2  # chord midpoint coordinates
        r = get_dist(0,0, Xm, Ym)  # distance between circle center and chord midpoint
        h = radius - r
        segment_area = radius**2 * math.acos((radius-h)/radius) - (radius-h) * math.sqrt(2*radius*h - h**2)

        included_area = triangle_area + segment_area
    
    else:  # no intersections, circle completely inside or completely outside
        if mesh.center_in_box(x, y, vertex, side):
            included_area = math.pi * radius**2
        else:
            included_area = 0.0

    return included_area


def separate_circles(rand_gen, w, x0, y0, x1, y1):
    x0 = x0 + ((x0-x1)* w * rand_gen.random()) / abs(x0-x1)
    y0 = y0 + ((y0-y1)* w * rand_gen.random()) / abs(y0-y1)
    return x0, y0


def generate_rve(rand_gen, w, Vf, radius, vertex, side, min_dist, max_iter):

    number = (Vf * side**2) / (math.pi * radius**2)

    centers = get_positions(
        rand_gen, number, radius, vertex, side, min_dist, max_iter,
    )
    target_area = Vf * side**2
    total_area = 0
    flag = False

    logger.info("initial number of centers: %s", len(centers))
    # logger.debug("initial centers: %s", centers)

    while (total_area <= target_area) or (flag == False):
        total_area = 0.0
        if flag == True:
            additional_centers = get_positions(
                rand_gen, 2, radius, vertex, side, min_dist, max_iter,
            )
            centers.extend(additional_centers)

        flag = True
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i != j:
                    dist = get_dist_squared(centers[i][0], centers[i][1], centers[j][0], centers[j][1])
                    if dist <= min_dist:
                        flag = False
                        centers[i][0], centers[i][1] = separate_circles(
                            rand_gen, w, centers[i][0], centers[i][1], centers[j][0], centers[j][1],
                        )
            included_area = get_included_area(
                centers[i][0], centers[i][1], radius, vertex, side,
            )
            total_area += included_area
        logger.debug("flag: %s", flag)
        logger.debug("total area: %s", total_area)
        logger.debug("target area: %s\n", target_area)
    return centers


if __name__ == "__main__":

    rand_gen = np.random.default_rng(96)
    Vf = 0.60
    # number = 15
    radius = 1.0
    vertex = [0.0, 0.0, 0.0]
    side = 50
    min_dist_square = (2.1 * radius)**2
    max_iter = 100000
    
    start_time = time.time()
    centers = ge_wang_rve(rand_gen, w, Vf, radius, vertex, side, min_dist_square, max_iter)
    # print(len(centers))
    print(centers)
    print(f"--- {time.time() - start_time} seconds ---")

    centers = mesh.filter_centers(centers, radius, vertex, side)

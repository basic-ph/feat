from feat import ge_wang
import numpy as np



def test_get_circle_side_intersection():
    radius = 1.0

    x = -0.5; y = 3.0
    x1 = 0.0; y1 = 0.0
    x2 = 0.0; y2 = 5.0
    output1 = ge_wang.get_circle_side_intersection(x, y, radius, x1, y1, x2, y2)
    xa = output1[0]+x; ya = output1[1]+y
    xb = output1[2]+x; yb = output1[3]+y
    output1 = [round(i,3) for i in [xa,ya,xb,yb]]

    x = 3.2; y = 4.8
    x1 = 0.0; y1 = 5.0
    x2 = 5.0; y2 = 5.0
    output2 = ge_wang.get_circle_side_intersection(x, y, radius, x1, y1, x2, y2)
    xa = output2[0]+x; ya = output2[1]+y
    xb = output2[2]+x; yb = output2[3]+y
    output2 = [round(i,3) for i in [xa,ya,xb,yb]]

    # corner circle
    x = 5.4; y = -0.2
    x1 = 0.0; y1 = 0.0
    x2 = 5.0; y2 = 0.0
    output3 = ge_wang.get_circle_side_intersection(x, y, radius, x1, y1, x2, y2)
    xa = output3[0]+x; ya = output3[1]+y
    output3 = [round(i,3) for i in [xa,ya]]
    
    x = 5.4; y = -0.2
    x1 = 5.0; y1 = 0.0
    x2 = 5.0; y2 = 5.0
    output4 = ge_wang.get_circle_side_intersection(x, y, radius, x1, y1, x2, y2)
    xa = output4[0]+x; ya = output4[1]+y
    output4 = [round(i,3) for i in [xa,ya]]

    x = 6.5; y = 4.2
    x1 = 5.0; y1 = 5.0
    x2 = 0.0; y2 = 5.0
    output5 = ge_wang.get_circle_side_intersection(x, y, radius, x1, y1, x2, y2)

    assert output1 == [0.0, 3.866, 0.0, 2.134]
    assert output2 == [4.18, 5.0, 2.22, 5.0]
    assert output3 == [4.42, 0.0]
    assert output4 == [5.0, 0.717]
    assert output5 == None


def test_get_included_area():
    x = -0.5; y = 3.0
    radius = 1.0
    vertex = [0.0, 0.0, 0.0]
    side = 5.0
    area1 = ge_wang.get_included_area(x, y, radius, vertex, side)
    
    x = 3.2; y = 4.8
    area2 = ge_wang.get_included_area(x, y, radius, vertex, side)
    
    x = 5.4; y = -0.2
    area3 = ge_wang.get_included_area(x, y, radius, vertex, side)

    x = 2.5; y = 3.0
    area4 = ge_wang.get_included_area(x, y, radius, vertex, side)

    assert round(area1, 3) == 0.614
    assert round(area2, 3) == 1.968
    assert round(area3, 3) == 0.278
    assert round(area4, 3) == 3.142


def test_separate_circles():
    rand_gen = rand_gen = np.random.default_rng(96)
    Vf = 0.5
    w = -11.5 * Vf**2 - 4.3*Vf + 8.5

    x0 = 3.0; y0 = 3.0
    x1 = 2.0; y1 = 2.0

    x0_new, y0_new = ge_wang.separate_circles(rand_gen, w, x0, y0, x1, y1)
    # output: 5.533862352324405 5.550266371277898
    print(x0_new, y0_new)
    assert x0_new > x0
    assert y0_new > y0


"""
    while (total_area < (Vf * side**2)) or (flag == False):
    if flag == True:
        # add new circle position
        pass
    flag = True

    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j: # diverso da

                if distance(i,j) <= l_overlap:
                    flag = False
                    # move I away from J (equation (3))
                    pass
        
        total_area += area(i) # solo dentro al dominio
"""
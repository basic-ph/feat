// 
r = 0.5;
hole_num = 3;
cl = 1;
cl2 = 1; // hole carat. lenght
side = 10; // side: box side`


x_array[] = {1, 5, 7};
y_array[] = {3, 6, 9};



// BOX 
Point(1) = {0, 0, 0, lc};
Point(2) = {side, 0, 0, lc};
Point(3) = {side, side, 0, lc};
Point(4) = {0, side, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Macro Test

    p1 = newp; Point(p1) = {x+r, y, z, lch};
    p2 = newp; Point(p2) = {x, y, z, lch};
    p3 = newp; Point(p3) = {x-r, y, z, lch};

    c1 = newreg; Circle(c1) = {p1, p2, p3};
    c2 = newreg; Circle(c2) = {p3, p2, p1};

    loop_list[i] = newreg;
    Curve Loop(loop_list[i]) = {c1, c2}; // one complete hole

    hole = newreg;
    Plane Surface(hole) = {loop_list[i]};

Return


x = 5; y = 5; z = 0; r = 0.5;

For i In {0:2}

    x = x_array[i];
    y = y_array[i];

    Call Test ;
    Physical Surface(i) = hole; // mi servono??

    // Printf("Hole %g (center = {%g,%g,%g}, radius = %g) has number %g!", i, x, y, z, r, hole);

EndFor

loop_list[0] = newreg; // save tag for box as first item in list
Curve Loop(loop_list[0]) = {1, 2, 3, 4};

phy = newreg;
Plane Surface(phy) = {loop_list[]};

Physical Surface("cheese", newreg) = phy ;
Physical Curve("left side", newreg) = {4}; // dirichlet
Physical Curve("right side", newreg) = {2}; // neumann
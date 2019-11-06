Include "header.geo";


Macro Test

    p1 = newp; Point(p1) = {x+r, y, 0, hcl};
    p2 = newp; Point(p2) = {x, y, 0, hcl};
    p3 = newp; Point(p3) = {x-r, y, 0, hcl};

    c1 = newreg; Circle(c1) = {p1, p2, p3};
    c2 = newreg; Circle(c2) = {p3, p2, p1};

    loop_list[i] = newreg;
    Curve Loop(loop_list[i]) = {c1, c2}; // one complete hole

    // hole = newreg;
    // Plane Surface(hole) = {loop_list[i]};

Return


// BOX 
Point(1) = {0, 0, 0, cl};
Point(2) = {side, 0, 0, cl};
Point(3) = {side, side, 0, cl};
Point(4) = {0, side, 0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// HOLES
For i In {1:hole_num}

    x = x_array[i-1];
    y = y_array[i-1];

    Call Test ;
    // Physical Surface(i) = hole; // mi servono??

EndFor

loop_list[0] = newreg; // save tag for box as first item in list
Curve Loop(loop_list[0]) = {1, 2, 3, 4};

phy = newreg;
Plane Surface(phy) = {loop_list[]};

Physical Surface("cheese", newreg) = phy ;
Physical Curve("left side", newreg) = {4}; // dirichlet
Physical Point("bottom left corner", newreg) = {1}; // dirichlet
Physical Curve("right side", newreg) = {2}; // neumann

Mesh 2;
Save "../msh/common.msh";
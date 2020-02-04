// feap_2

cl = 2;

Point(1) = {0.0, 0.0, 0.0, cl};
Point(2) = {1.3, 0.7, 0.0, cl};
Point(3) = {0.0, 1.0, 0.0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};


Physical Surface("rubber", 1) = {1};
Physical Curve("left side", 2) = {3};
Physical Point("bottom corner", 3) = {1};
Physical Point("right corner", 4) = {2};

Mesh 2;
Save "../msh/feap_2.msh";
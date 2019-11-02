lc = 2;

Point(1) = {0, 0, 0, lc};
Point(2) = {2, 0, 0, lc};
Point(3) = {2, 2, 0, lc};
Point(4) = {0, 2, 0, lc};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("left side", 1) = {4}; // dirichlet
Physical Curve("right side", 2) = {2}; // neumann
Physical Surface("dummy") = {1}; // group indicating dummy material

Mesh 2;
Save "..\msh\test_k.msh";
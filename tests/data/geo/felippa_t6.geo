// Test on Straight-Sided Triangle
// from section 24.4.3 of Introduction to Finite Element Methods (ASEN 5007) - Carlos Felippa
// see http://kis.tu.kielce.pl//mo/COLORADO_FEM/colorado/Home.html

cl = 10;

Point(1) = {0, 0, 0, cl};
Point(2) = {6, 2, 0, cl};
Point(3) = {4, 4, 0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};

Physical Surface("Be") = {1};

Mesh.ElementOrder = 2;
// Mesh.HighOrderOptimize = 2;

Mesh 2;
Save "../msh/felippa_t6.msh";

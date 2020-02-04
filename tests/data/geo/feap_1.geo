// feap_1

cl = 3;

// Element 1 
Point(1) = {0, 0, 0, cl};
Point(2) = {2, 0, 0, cl};
Point(3) = {2, 2, 0, cl};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};

// Element 2
Point(4) = {0, 2, 0, cl};
Line(4) = {3, 4};
Line(5) = {4, 1};
Curve Loop(2) = {4, 5, -3};
Plane Surface(2) = {2};

// Physical Groups
Physical Surface("cheese", 1) = {1, 2};
Physical Curve("left side", 2) = {5}; // dirichlet
Physical Point("bottom left corner", 3) = {1};  // dirichlet
Physical Curve("right side", 4) = {2}; // neumann


Mesh 2;
Save "../msh/feap_1.msh";
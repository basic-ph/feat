cl = 5;

// ELEMENT 0
Point(1) = {0, 0, 0, cl};
Point(2) = {3, 0, 0, cl};
Point(3) = {3, 2, 0, cl};
Point(4) = {0, 2, 0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface("steel", newreg) = {1};  // material
Physical Curve("left side", newreg) = {4}; // dirichlet
Physical Point("bottom right corner", newreg) = {2}; // dirichlet
Physical Point("top right corner", newreg) = {3}; // neumann

Mesh 2;
Save "../msh/test.msh";
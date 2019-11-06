// Include "header.geo";

side = 2;
cl = 2;

// BOX 
Point(1) = {0, 0, 0, cl};
Point(2) = {side, 0, 0, cl};
Point(3) = {side, side, 0, cl};
Point(4) = {0, side, 0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface("cheese", newreg) = {1} ;
Physical Curve("left side", newreg) = {4}; // dirichlet
Physical Point("bottom left corner", newreg) = {1}; // dirichlet
Physical Curve("right side", newreg) = {2}; // neumann

Mesh 2;
Save "../msh/base.msh";
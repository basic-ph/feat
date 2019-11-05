import numpy as np

# SETTINGS
r = 0.5
hole_num = 10
cl = 1
hcl = 0.5
side = 10

min_dist = 3 * r
offset = 1


rg = np.random.default_rng()  # accept seed as arg

dist = lambda x_0, y_0, x_1, y_1: np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)

x_array = np.zeros(hole_num)
y_array = np.zeros(hole_num)
valid = np.ones(hole_num, dtype=np.int16)

i = 0

while True:

    print("i:", i)

    x = offset + (side - 2*offset)* rg.random()
    y = offset + (side - 2*offset)* rg.random()

    for j in range(i):
        distance = dist(x, y, x_array[j], y_array[j])
        if distance > min_dist:
            valid[j] = 1
        else:
            valid[j] = 0

    print("valid", valid)
    
    if 0 in valid:
        pass
    else:
        x_array[i] = x
        y_array[i] = y
        i += 1

    if i == hole_num:
        break


print(x_array)
print(y_array)

nl = "\n"
final_curly = "\n};\n" 
header = (
    f"//{nl}"
    f"r = {r};{nl}"
    f"hole_num = {hole_num};{nl}"
    f"cl = {cl};{nl}"
    f"hcl = {hcl};{nl}"
    f"side = {side};{nl}"
    f"{nl}"
    f"{nl}"
)

x_string = "x_array[] = {\n    "
for k in range(hole_num):
    if k == hole_num-1:
        x_string += f"{x_array[k]}"
    else:
        x_string += f"{x_array[k]}, "
   
x_string += final_curly

y_string = "y_array[] = {\n    "
for k in range(hole_num):
    if k == hole_num-1:
        y_string += f"{y_array[k]}"
    else:
        y_string += f"{y_array[k]}, "
   
y_string += final_curly

with open("../gmsh/geo/header.geo", "w") as geo:
    geo.write(header)
    geo.write(x_string)
    geo.write(y_string)
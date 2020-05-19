import pygmsh

def main():

    coarse_cl = 0.5
    fine_cl = 0.1
    radius = 1.0

    geom = pygmsh.opencascade.Geometry()

    centers = [
        [2.5, 2.5, 0.0],
        [5.0, 5.0, 0.0],
        [7.5, 5.0, 0.0],
        [3.0, 7.0, 0.0],
    ]

    disks = []
    for i in range(4):
        fiber =  geom.add_disk(centers[i], 1.0)
        disks.append(fiber)
    # fiber1 = geom.add_disk(centers[0], 1.0)
    # fiber2 = geom.add_disk(centers[1], 1.0)
    # fiber3 = geom.add_disk(centers[2], 1.0)
    # fiber4 = geom.add_disk(centers[3], 1.0)
    # disks = [fiber1, fiber2, fiber3, fiber4]

    print(disks)
    disk_tags = ", ".join([d.id for d in disks])
    print(disk_tags)

    x0 = [2.5, 2.5, 0.0]
    side = 5.0
    rectangle = geom.add_rectangle(x0, side, side)

    # inter = geom.boolean_intersection([rectangle,fiber1, fiber2], delete_first=True, delete_other=False)
    # diff = geom.boolean_difference([rectangle], disks, delete_first=True, delete_other=False)
    geom.add_raw_code(
        f"BooleanIntersection{{ Surface{{{disk_tags}}}; Delete; }} "
        f"{{ Surface{{{rectangle.id}}}; }}"
    )
    geom.add_raw_code(
        f"BooleanDifference{{ Surface{{{rectangle.id}}}; Delete; }} "
        f"{{ Surface{{{disk_tags}}}; }}"
    )

    e = 0.01
    geom.add_raw_code(
        f"p() = Point In BoundingBox"
        f"{{{x0[0]-e}, {x0[1]-e}, {x0[2]-e}, {x0[0]+e}, {x0[1]+e}, {x0[2]+e}}};"  # bottom left corner
    )
    geom.add_raw_code(
        f"q() = Curve In BoundingBox"
        f"{{{x0[0]-e}, {x0[1]-e}, {x0[2]-e}, {x0[0]+e}, {side+e}, {x0[2]+e}}};"  # left side
    )
    geom.add_raw_code(
        f"r() = Curve In BoundingBox"
        f"{{{side-e}, {x0[1]-e}, {x0[2]-e}, {side+e}, {side+e}, {x0[2]+e}}};"  # right side
    )

    geom.add_raw_code(
        f"boundary[] = Boundary{{ Surface{{{disk_tags}}}; }};"
    )

    geom.add_raw_code("Field[1] = Distance;")
    geom.add_raw_code("Field[1].NNodesByEdge = 100;")
    geom.add_raw_code(f"Field[1].EdgesList = {{boundary[]}};")
    geom.add_raw_code(
        "Field[2] = Threshold;\n"
        "Field[2].IField = 1;\n"
        f"Field[2].LcMin = {fine_cl};\n"
        f"Field[2].LcMax = {coarse_cl};\n"
        "Field[2].DistMin = 0.01;\n"
        f"Field[2].DistMax = {radius};\n"  # FIXME test this
        "Background Field = 2;"
    )
    geom.add_raw_code(
        "Mesh.CharacteristicLengthExtendFromBoundary = 0;\n"
        "Mesh.CharacteristicLengthFromPoints = 0;\n"
        "Mesh.CharacteristicLengthFromCurvature = 0;"
    )

    geom.add_raw_code(
        f"Physical Surface(\"matrix\") = {{{rectangle.id}}};\n"
        f"Physical Surface(\"fiber\") = {{{disk_tags}}};\n"
        f"Physical Point(\"bottom left corner\") = {{p()}};\n"
        f"Physical Line(\"left side\") = {{q()}};"
        f"Physical Line(\"right side\") = {{r()}};\n"
    )
    
    mesh = pygmsh.generate_mesh(
        geom,
        geo_filename="../data/geo/cake.geo",  # uncomment this for saving geo and msh
        # msh_filename=str(msh_path),
        verbose=True,
        dim=2,
    )

if __name__ == "__main__":
    main()
import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import fem
from feat import mesh


def main():
    # argument parser creation and setup
    desc = (
        "Complete an RVE analysis for the evaluation of transverse modulus "
        "of unidirectional fiber-reinforced composite materials"
    )
    parser = argparse.ArgumentParser(
        prog="python rve.py",
        description=desc,
    )
    parser.add_argument(
        "version",
        help="specify which version of the FEA code should be used",
        choices=["base", "sparse", "vector"],
        default="vector",
        nargs="?"
    )
    args = parser.parse_args()

    if args.version == "base":
        analysis = fem.base_analysis
    elif args.version == "sparse":
        analysis = fem.sp_base_analysis
    else:  # if no version arg is provided the default is used
        analysis = fem.vector_analysis

    # DATA
    Vf = 0.30  # fiber volume fraction
    radius = 1.0  # fiber radius
    min_distance = 2.1 * radius
    offset = 1.1 * radius
    max_iter = 100000
    coarse_cls = [0.5, 0.25, 0.12, 0.06]  #  [1.0, 0.5, 0.25, 0.12, 0.06] coarse element dimension (far from matrix-fiber boundary)
    fine_cls = [cl / 2 for cl in coarse_cls]  # fine element dimension (matrix-fiber boundary)
    element_type = "triangle"

    max_number = 1000
    max_side = math.sqrt(math.pi * radius**2 * max_number / Vf)
    logger.info("-------- RVE ANALYSIS --------")
    logger.info("max number: %s - max side: %s", max_number, max_side)

    num_steps = 10 # number of steps from the
    side_step = max_side / ((num_steps+1)*2)  # distance between box vertices of different RVE

    seeds = [96, 11, 50, 46, 88, 66, 89, 15, 33, 49]  # [96, 11, 50, 46, 88, 53, 89, 15, 33, 49]
    num_samples  = 6  # can't exceed seeds lenght
    data = []  # list of [s: id del sample, n: numero di fibre nel dominiio, coarse_cl, side, num_nodes, E2, {0|1}]

    for p in range(num_samples):
        rand_gen = np.random.default_rng(seeds[p])
        centers = []
        centers = mesh.get_fiber_centers(rand_gen, max_number, max_side, min_distance, offset, max_iter, centers)
        # logger.debug("centers:\n%s", centers)

        for s in range(num_steps):
            r = num_steps - s  # reversing succession, from small to large RVE
            box_vertex = [r*side_step, r*side_step, 0.0]
            box_side = max_side - (r*2*side_step)
            filtered_centers = mesh.filter_centers(centers, radius, box_vertex, box_side)
            moduli = []  # clean list used for mesh convergence validation
            # logger.info("filtered centers:\n%s", fkiltered_centers)

            for m in range(len(coarse_cls)):
                filename = f"rve-{p}-{s}-{m}"
                geo_path = "../data/happy/" + filename + ".geo"
                msh_path = "../data/happy/" + filename + ".msh"
                coarse_cl = coarse_cls[m]
                fine_cl = fine_cls[m]
                mesh_obj = mesh.create_mesh(
                    geo_path,
                    msh_path,
                    radius,
                    box_vertex,
                    box_side,
                    filtered_centers,
                    coarse_cl,
                    fine_cl
                )
                num_nodes = mesh_obj.points.shape[0]
                # run FEM simulation
                E2 = analysis(mesh_obj, element_type, post_process=True, vtk_filename=filename)
                logger.info("SAMPLE %s - STEP %s - MESH %s - nodes: %s - E2: %s", p, s, m, num_nodes, E2)
                
                if m == 0:  # first mesh
                    moduli.append(E2) # store the value obtained for mesh convergence validation
                    data.append([p, s, m, box_side, num_nodes, E2, 0])  # 0 means non-converged result
                else:
                    moduli.append(E2)
                    prev_E2 = moduli[m-1]
                    rel_diff = abs(E2 - prev_E2) / prev_E2  # difference relative to precedent obtained estimate
                    if rel_diff < 0.002:  # 0.2%
                        data.append([p, s, m, box_side, num_nodes, E2, 1])  # 1 means converged result
                        logger.info("SAMPLE %s - STEP %s - MESH %s - converged!\n", p, s, m)
                        break  # mesh convergence obtained, continue with the next random realization
                    else:
                        data.append([p, s, m, box_side, num_nodes, E2, 0])  # 0 means non-converged result
                        logger.info("SAMPLE %s - STEP %s - MESH %s - NOT converged!\n", p, s, m)


    actual_date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    data_file = f"../data/csv/{actual_date}.csv"
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
    logger.info("Output written to: %s", data_file)



if __name__ == "__main__":
    # LOGGING (you can skip this)
    log_lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    handler = logging.StreamHandler()
    handler.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    start_time = time.time()
    main()
    print(f"--- {time.time() - start_time} seconds ---")

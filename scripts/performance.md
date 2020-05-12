# Performance data - 2020-04-30
The following table reports timings for various implementations of the FEM code.  

Domain/Mesh |num nodes | base | sparse | vector |
----------- | -------- | ---- | ------ | ------ |
perf 25 | 8826 | 21.688550233840942 | 1.0718538761138916 | 0.3392636775970459 |
perf 50 | 17820 | 169.10492253303528 | 2.2305076122283936 | 0.7658224105834961 |
perf 100 | 35327 | -- | 4.581058979034424 | 1.6824288368225098 |
perf 400 | 141282 | -- | 19.046363592147827 | 7.945904970169067 |
perf 1600 | 565549 | -- | 82.52184128761292 | 40.08019495010376 |

# cProfile performance test
perf_50 mesh for every variant.
 - perf_50_base.prof
 - perf_50_sparse.prof
 - perf_50_vector.prof

Data summarized:
- base_analysis,                        192.858 s (100%)
    - numpy.linalg.solve                190.158 s (%)
    - assembly                          2.437 s (%)
    - apply_dirichlet                   0.207 s (%)
 
- sp_base_analysis                      2.552 s
    - sp_assebly                        2.212 s
    - sparse.linalg.spsolve             0.256 s
    - sp_apply_dirichlet                0.035 s

- vector_analysis REDO!!!               0.706 s
    - assembly                          0.399 s
    - sparse.linalg.spsolve             0.258 s
    - sp_apply_dirichlet                0.034 s

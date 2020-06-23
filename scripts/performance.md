# Performance data - 2020-04-30
The following table reports timings for various implementations of the FEM code.  

Domain/Mesh |num nodes | base | sparse | vector |
----------- | -------- | ---- | ------ | ------ |
perf 25 | 8826 | 21.688550233840942 | 1.0718538761138916 | 0.3392636775970459 |
perf 50 | 17820 | 169.10492253303528 | 2.2305076122283936 | 0.7658224105834961 |
perf 100 | 35327 | -- | 4.581058979034424 | 1.6824288368225098 |
perf 400 | 141282 | -- | 19.046363592147827 | 7.945904970169067 |
perf 1600 | 565549 | -- | 82.52184128761292 | 40.08019495010376 |


## 2020-06-17
perf 25 | base | 22.343284845352173  
perf 50 | base | 184.2389760017395  

perf 25 | sparse | 1.0877270698547363  
perf 50 | sparse | 2.205213785171509  
perf 100 | sparse | 4.466970205307007  
perf 400 | sparse | 19.1698796749115  
perf 1600 | sparse | 83.22076272964478  

perf 25 | vector | 0.25047993659973145  
perf 50 | vector | 0.5708961486816406  
perf 100 | vector | 1.258349895477295  
perf 400 | vector | 5.730280876159668  
perf 1600 | vector | 29.83337163925171  


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


RVE analysis
- 500 fibre, 10 samples, 10 steps, seeds = [44, 5, 34, 58, 11, 16, 91, 77, 84, 11]
  tempo = 5797.039974927902 s = 96.617' ~ 1h36'
- 500 fibre, 5 samples, 10 steps, seeds = [24, 21, 65, 22, 35]
  tempo = 
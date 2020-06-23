# FEAT (Finite Element Acceleration Techniques)

## conda
```
conda create -n feat python=3.8 numpy scipy snakeviz pytest
conda activate feat
conda install -c conda-forge meshio
```
per generare il file YAML:
```
conda env export --from-history > environment.yml
conda env export > environment.yml
```
per ricreare l'ambiente usando il file .yml usare:
```
conda env create -f environment.yml
```

## pytest
```
pip install -e .
```
installa feat come pacchetto in *editable* mode.  
Necessario perché `pytest` funzioni correttamente, altrimenti va 
utilizzato con il comando ```python -m pytest``` per eseguire i test direttamente 
contro la copia locale senza usare pip.  

## profile
from: [snakeviz docs](https://jiffyclub.github.io/snakeviz/#generating-profiles)  
```
python -m cProfile -o program.prof my_program.py
```

## memory-profiler
commands:
    - `mprof run`: running an executable, recording memory usage
    - `mprof plot`: plotting one the recorded memory usage (by default, the last one)
    - `mprof list`: listing all recorded memory usage files in a user-friendly way.
    - `mprof clean`: removing all recorded memory usage files.
    - `mprof rm`: removing specific recorded memory usage files

## ~~GMSH python API~~
from:
https://gitlab.onelab.info/gmsh/gmsh/blob/master/demos/api/README.txt#L19  
>To run the Python examples, add the "lib" directory from the SDK to PYTHONPATH,  
e.g., if you are currently in the root directory of the SDK:  
>   export PYTHONPATH=${PYTHONPATH}:${PWD}/lib
>then run e.g.  
>   python share/doc/gmsh/demos/api/t1.py

# FEAT (Finite Element Acceleration Techniques)

## pytest
```
pip install -e .
```
installa feat come pacchetto in *editable* mode.  
Necessario perché `pytest` funzioni correttamente, altrimenti va 
utilizzato con il comando ```python -m pytest``` per eseguire i test direttamente 
contro la copia locale senza usare pip.  


## conda
```
conda create -n feat python=3.7 numpy pytest
...
conda activate feat
pip install meshio

... OK
```
per ricreare l'ambiente usando il file .yml usare:
NON FUNZIONANTE (riprovare?)
```
conda env create --file envname.yml
```
per generare il file YAML:
```
conda env export --name feat > environment.yml
```

## GMSH python API
from:
https://gitlab.onelab.info/gmsh/gmsh/blob/master/demos/api/README.txt#L19  
>To run the Python examples, add the "lib" directory from the SDK to PYTHONPATH,  
e.g., if you are currently in the root directory of the SDK:  
>   export PYTHONPATH=${PYTHONPATH}:${PWD}/lib
>then run e.g.  
>   python share/doc/gmsh/demos/api/t1.py

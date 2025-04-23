Le repo se découpe en différentes parties. Deux notebooks orientés visualisation des résultats : `vizu.ipynb` pour visualiser les images originales et `vizu_cv.ipynb` qui permet de visualiser les résultats partagés via le Filesender.

Ensuite, cette partie n'a pas besoin d'être relancé sauf recalcul des résultats. Le notebook `src/ACOUSUR_demo.ipynb` est une adaptation du notebook `Tuto_REACTIV_POLAR.ipynb` du repo `Reactiv`. 
Une adaptation du pipeline sous forme de script est disponible en version gpu :

```
python src/run_pipeline.py
```

ou en version cpu : 

```
python src/run_pipeline_cpu.py
```

A priori, il n'est pas nécessaire de run de nouveau ce script, les résultats ayant été partagés via le Filesender Renater.

Un script pour obtenir les stacks des images est disponible aussi : 

```
python src/stack_images.py
```


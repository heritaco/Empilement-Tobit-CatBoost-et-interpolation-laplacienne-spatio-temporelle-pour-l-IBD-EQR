En esta notebook se propone una forma de limpiar la wida db porque eran muchos datos. 

```
path: data/processed/clean_df.parquet
```

Luego se obtiene solo un modelo de Random Forest: 
```
path: results\flirting w models\01 random forest for 1 ibd.joblib
```

```python
Mean Squared Error: 0.7238849809788043
Mean Absolute Error: 0.5332663107609152
R2 Score: 0.9097291236593544
```

![alt text](images/result.png)

Asi se carga el modelo:

```python
import joblib

clf = joblib.load('../../results/flirting w models/01 random forest for 1 ibd.joblib')
clf
```

Este modelo creo que es malo, no estoy seguro si podria mejorar haciendo muchas regresiones por cada hydroecoregion, porque el IBD se ven como 3 regresiones diferentes.

![alt text](images/grafica_ibd.png)

Esta regresion es unica, entonces una mejor regresion serian 3 regresiones diferentes.

O podemos forzar a overfittear el modelo? Como, haciendo que aprenda de sus errores? jajaaajajaj
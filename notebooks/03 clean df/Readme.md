Solo nos quedamos con las columnas que tengan al menos 95% de datos no nulos.
De las columnas que quitamos nos quedamos con la suma de ellas y las guardamos como "uncommon_taxons".

Nos quedamos con 171 columnas.
En 03_CLEAN_COMPLETE_DF.parquet tenemos el dataframe limpio con las columnas de IBD, IBD_EQR e IBD_EQR_Status, las que tienen NaN en IBD son las que usaremos para predecir y las que tienen algo en IBD son las que usaremos para entrenar.
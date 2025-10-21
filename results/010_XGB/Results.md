XBG on all data: 49.99%
XGB on cleaned data: 37.23%

Creo que estamos quitando demasiada información relevante al limpiar los datos. Podríamos intentar ajustar los parámetros del modelo o explorar diferentes técnicas de limpieza que conserven más características importantes. También sería útil analizar qué tipo de datos se están eliminando durante la limpieza para entender mejor su impacto en el rendimiento del modelo.




Para el `XGB Trucated interpolated_full_2.csv`
```
        reg_params = dict(
            # objective="survival:aft",
            # eval_metric="aft-nloglik",
            # aft_loss_distribution="normal",                 # try "logistic" if heavy tails
            # aft_loss_distribution_scale=0.8,                # tune 0.4–1.5
            # tree_method="hist",
            max_depth=8,
            max_leaves=0,                                   # 0 lets xgb choose
            learning_rate=0.03,
            n_estimators=2500,                              # use early stopping
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=2.0,
            max_bin=512
        )
```


Para el `XGB Trucated interpolated_full_3.csv`
```
        reg_params = dict(
            # objective="survival:aft",
            # eval_metric="aft-nloglik",
            # aft_loss_distribution="normal",                 # try "logistic" if heavy tails
            # aft_loss_distribution_scale=0.8,                # tune 0.4–1.5
            # tree_method="hist",
            max_depth=8,
            max_leaves=0,                                   # 0 lets xgb choose
            learning_rate=0.03,
            n_estimators=5000,                              # use early stopping
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=2.0,
            max_bin=512
        )
```
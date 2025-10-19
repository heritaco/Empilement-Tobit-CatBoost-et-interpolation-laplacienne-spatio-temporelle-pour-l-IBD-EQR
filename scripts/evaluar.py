from __future__ import annotations
import os
import math
import pandas as pd
from typing import List, Tuple, Dict, Optional

def contra_modelos(
    df_nuevo: pd.DataFrame,
    modelos: List[Tuple[str, float]],
    *,
    id_col: str = "SamplingOperations_code",
    pred_col: str = "IBD_EQR_Status",
    carpeta: Optional[str] = None,  # si todos están en la misma carpeta, puedes pasarla aquí
    candidatos_pred_cols: Optional[List[str]] = None,  # override si ya conoces el nombre exacto
    normalizar_labels: bool = True,
) -> Dict[str, object]:
    """
    Compara las predicciones de `df_nuevo` contra varios modelos ya evaluados (guardados como CSV)
    y contra consensos (mayoría simple y mayoría ponderada por accuracies). También estima la
    probabilidad esperada de que la predicción de `df_nuevo` sea correcta.

    Parámetros
    ----------
    df_nuevo : DataFrame
        Debe contener [id_col, pred_col] con las predicciones a evaluar.
    modelos : List[Tuple[str, float]]
        Lista de (ruta_csv, accuracy) para cada modelo histórico evaluado.
        - 'accuracy' debe venir en [0,1]. Si la tienes en %, pásala/convierte a proporción.
    id_col : str
        Nombre de la columna id. Default: "SamplingOperations_code".
    pred_col : str
        Nombre de la columna de la predicción de df_nuevo. Default: "IBD_EQR_Status".
    carpeta : str | None
        Carpeta base donde están los CSV (si la ruta de cada CSV no es absoluta).
    candidatos_pred_cols : list[str] | None
        Lista de posibles nombres de columna de predicción dentro de cada CSV.
        Si None, se intenta autodetección inteligente.
    normalizar_labels : bool
        Si True, hace strip y upper() en labels string para robustez.

    Retorna
    -------
    dict con:
      - 'resumen_modelos': DataFrame (modelo, accuracy, n_overlap, coincidencia, archivo)
      - 'acuerdo_consenso_simple': float
      - 'acuerdo_consenso_ponderado': float
      - 'esperado_correcto_promedio': float
      - 'detallado': DataFrame con columnas:
            [id_col, pred_col, <preds de cada modelo>, consenso_simple, consenso_pond,
             prob_correcta_estimada, match_<modelo>...]
    """
    # --- Validaciones básicas
    if id_col not in df_nuevo.columns or pred_col not in df_nuevo.columns:
        raise ValueError(f"`df_nuevo` debe contener las columnas '{id_col}' y '{pred_col}'.")

    df_base = df_nuevo[[id_col, pred_col]].copy()

    # Normalización suave de labels para evitar mismatches por casing/espacios
    def _norm(x):
        if pd.isna(x):
            return x
        if isinstance(x, str):
            s = x.strip()
            return s.upper() if normalizar_labels else s
        return x

    if normalizar_labels:
        df_base[pred_col] = df_base[pred_col].map(_norm)

    # Heurística de autodetección del nombre de columna de predicción
    default_candidates = [
        pred_col,
        "IBD_EQR_Status",
        "prediction",
        "pred",
        "status",
    ]
    # Permitimos columnas con prefijo, p.ej. IBD_EQR_Status_81
    def _find_pred_col(cols: List[str]) -> str:
        cands = (candidatos_pred_cols or default_candidates)
        # 1) coincidencia exacta por candidatos
        for c in cands:
            if c in cols:
                return c
        # 2) heurística: la que empiece con 'IBD_EQR_Status'
        for c in cols:
            if str(c).startswith("IBD_EQR_Status"):
                return c
        # 3) fallback: escoger la primera no-id con dtype 'object' o 'category'
        for c in cols:
            if c != id_col:
                return c
        raise ValueError("No pude detectar la columna de predicción del CSV del modelo.")

    # Cargar predicciones de modelos y armar tabla combinada
    tablas = []
    resumen_rows = []
    nombre_cols_modelo = []

    for ruta, acc in modelos:
        if acc > 1.0:  # si viene en %, conviértelo
            acc = acc / 100.0

        archivo = os.path.join(carpeta, ruta) if (carpeta and not os.path.isabs(ruta)) else ruta
        df_m = pd.read_csv(archivo)

        if id_col not in df_m.columns:
            raise ValueError(f"En '{archivo}' no existe la columna id '{id_col}'.")

        col_pred_m = _find_pred_col(df_m.columns.tolist())
        col_modelo = os.path.splitext(os.path.basename(archivo))[0]  # nombre corto desde el filename
        col_modelo_pred = f"pred__{col_modelo}"  # evitar colisiones

        tmp = df_m[[id_col, col_pred_m]].rename(columns={col_pred_m: col_modelo_pred})

        if normalizar_labels:
            tmp[col_modelo_pred] = tmp[col_modelo_pred].map(_norm)

        tablas.append(tmp)
        nombre_cols_modelo.append((col_modelo, col_modelo_pred, acc, archivo))

    # Merge incremental por id
    df_all = df_base.copy()
    for _, col_pred_m, _, _ in nombre_cols_modelo:
        # merge secuencial (left) para conservar todos los ids de df_nuevo
        df_all = df_all.merge(
            next(t for t in tablas if col_pred_m in t.columns),
            on=id_col, how="left"
        )

    # Coincidencia por modelo
    for col_modelo, col_pred_m, acc, archivo in nombre_cols_modelo:
        match_col = f"match__{col_modelo}"
        df_all[match_col] = (df_all[pred_col] == df_all[col_pred_m]) & df_all[pred_col].notna() & df_all[col_pred_m].notna()

        overlap = df_all[col_pred_m].notna().sum()
        coinc = df_all[match_col].mean() if overlap > 0 else float("nan")

        resumen_rows.append({
            "modelo": col_modelo,
            "accuracy_reportada": acc,
            "n_overlap": int(overlap),
            "coincidencia_con_df": float(coinc) if not math.isnan(coinc) else None,
            "archivo": archivo
        })

    resumen_modelos = pd.DataFrame(resumen_rows)

    # Consenso (mayoría simple: sin ponderar)
    cols_modelos_pred = [c for _, c, _, _ in nombre_cols_modelo]
    if len(cols_modelos_pred) == 0:
        raise ValueError("No se proporcionaron modelos para comparar.")

    def _mode_row(row_vals):
        # mode() fila a fila (ignorando NaNs). Si hay empate, devuelve el primero por orden.
        s = pd.Series(row_vals).dropna()
        if s.empty:
            return pd.NA
        return s.mode().iloc[0]

    df_all["consenso_simple"] = df_all[cols_modelos_pred].apply(_mode_row, axis=1)

    # Consenso ponderado por accuracies: etiqueta con suma de pesos (accuracies) más alta
    etiqueta_unica = set()
    # recolectar universo de etiquetas posibles (para robustez)
    for c in cols_modelos_pred:
        etiqueta_unica.update(df_all[c].dropna().unique().tolist())

    etiquetas = list(etiqueta_unica)

    pesos = {col_pred_m: acc for _, col_pred_m, acc, _ in nombre_cols_modelo}

    def _consenso_ponderado(row):
        # suma de accuracies por etiqueta propuesta
        best_label, best_w = (pd.NA, -1.0)
        for label in etiquetas:
            w = 0.0
            for col in cols_modelos_pred:
                val = row[col]
                if pd.isna(val): 
                    continue
                if val == label:
                    w += pesos[col]
            if w > best_w:
                best_label, best_w = label, w
        return best_label

    df_all["consenso_pond"] = df_all.apply(_consenso_ponderado, axis=1)

    # Acuerdo df vs consensos
    acuerdo_simple = (df_all[pred_col] == df_all["consenso_simple"]) & df_all[pred_col].notna() & df_all["consenso_simple"].notna()
    acuerdo_ponderado = (df_all[pred_col] == df_all["consenso_pond"]) & df_all[pred_col].notna() & df_all["consenso_pond"].notna()

    acuerdo_consenso_simple = acuerdo_simple.mean()
    acuerdo_consenso_ponderado = acuerdo_ponderado.mean()

    # Estimación de prob. de estar correcto por registro
    # Heurística: prob_correcta(id) = (suma de accuracies de modelos que coinciden con df_nuevo) / (suma de accuracies disponibles)
    sum_acc_total = sum(pesos.values()) if len(pesos) else 0.0

    def _prob_correcta_estimada(row):
        if pd.isna(row[pred_col]) or sum_acc_total == 0:
            return pd.NA
        agree_w = 0.0
        any_obs = False
        for col in cols_modelos_pred:
            val = row[col]
            if pd.isna(val): 
                continue
            any_obs = True
            if val == row[pred_col]:
                agree_w += pesos[col]
        if not any_obs:
            return pd.NA
        return agree_w / sum_acc_total

    df_all["prob_correcta_estimada"] = df_all.apply(_prob_correcta_estimada, axis=1)
    esperado_correcto_promedio = float(df_all["prob_correcta_estimada"].dropna().mean()) if df_all["prob_correcta_estimada"].notna().any() else None

    # Orden final de columnas “bonitas”
    orden_cols = [id_col, pred_col] + cols_modelos_pred + [f"match__{m}" for m, _, _, _ in nombre_cols_modelo] + ["consenso_simple", "consenso_pond", "prob_correcta_estimada"]
    detallado = df_all[orden_cols].copy()

    return {
        "resumen_modelos": resumen_modelos.sort_values("modelo").reset_index(drop=True),
        "acuerdo_consenso_simple": float(acuerdo_consenso_simple) if acuerdo_consenso_simple == acuerdo_consenso_simple else None,
        "acuerdo_consenso_ponderado": float(acuerdo_consenso_ponderado) if acuerdo_consenso_ponderado == acuerdo_consenso_ponderado else None,
        "esperado_correcto_promedio": esperado_correcto_promedio,
        "detallado": detallado
    }

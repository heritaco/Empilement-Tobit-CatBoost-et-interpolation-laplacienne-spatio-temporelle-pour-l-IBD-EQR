import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================
print("📂 Cargando datos...")
ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\notebooks\06_cb_regression\completo\tp_full.parquet"
df = pd.read_parquet(ruta_archivo)

print(f"✓ Dataset cargado: {df.shape}")
print(f"  - Filas con IBD_EQR_Status: {df['IBD_EQR_Status'].notna().sum()}")
print(f"  - Filas a predecir: {df['IBD_EQR_Status'].isna().sum()}")

# Separar train (con target) y test (sin target)
train_df = df[df['IBD_EQR_Status'].notna()].copy()
test_df = df[df['IBD_EQR_Status'].isna()].copy()

print(f"\n📊 Distribución de clases en train:")
print(train_df['IBD_EQR_Status'].value_counts())

# ============================================
# 2. FEATURE ENGINEERING ESPACIAL
# ============================================
print("\n🗺️  Creando features espaciales...")

def crear_features_espaciales(df_source, df_target, k=5):
    """
    Crea features basadas en vecinos cercanos espacialmente
    """
    # Obtener coordenadas válidas
    coords_cols = ['Longitude_Lambert93', 'Latitude_Lambert93']
    valid_mask = df_source[coords_cols].notna().all(axis=1)
    
    if valid_mask.sum() == 0:
        return df_target
    
    coords_source = df_source.loc[valid_mask, coords_cols].values
    coords_target = df_target[coords_cols].values
    
    # Encontrar k vecinos más cercanos
    nbrs = NearestNeighbors(n_neighbors=min(k, len(coords_source)), metric='euclidean')
    nbrs.fit(coords_source)
    distances, indices = nbrs.kneighbors(coords_target)
    
    # Features de vecindad
    df_target['mean_distance_neighbors'] = distances.mean(axis=1)
    df_target['min_distance_neighbor'] = distances.min(axis=1)
    df_target['std_distance_neighbors'] = distances.std(axis=1)
    
    # Si tenemos target en source, crear features de consenso
    if 'IBD_EQR_Status' in df_source.columns:
        valid_source = df_source[valid_mask].reset_index(drop=True)
        
        for i, idx_list in enumerate(indices):
            neighbor_statuses = valid_source.iloc[idx_list]['IBD_EQR_Status'].values
            # Moda de los vecinos
            if len(neighbor_statuses) > 0:
                df_target.loc[df_target.index[i], 'neighbor_mode_status'] = \
                    pd.Series(neighbor_statuses).mode()[0] if len(pd.Series(neighbor_statuses).mode()) > 0 else 'Good'
    
    return df_target

# Aplicar features espaciales
train_df = crear_features_espaciales(train_df, train_df, k=10)
test_df = crear_features_espaciales(train_df, test_df, k=10)

# ============================================
# 3. SELECCIÓN Y LIMPIEZA DE FEATURES
# ============================================
print("\n🧹 Limpieza y selección de features...")

# Identificar columnas por tipo
id_cols = ['SamplingOperations_code']
target_cols = ['IBD', 'IBD_EQR', 'IBD_EQR_Status']
spatial_cols = ['Longitude_Lambert93', 'Latitude_Lambert93', 'Altitude']

# Columnas numéricas (especies y variables continuas)
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in target_cols and col not in spatial_cols]

# Columnas categóricas
categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                    if col not in id_cols and col not in target_cols]

print(f"  - Variables numéricas: {len(numeric_cols)}")
print(f"  - Variables categóricas: {len(categorical_cols)}")

# Eliminar features con más del 80% de NaNs
threshold_nan = 0.8
cols_to_keep = []

for col in numeric_cols:
    nan_ratio = train_df[col].isna().sum() / len(train_df)
    if nan_ratio < threshold_nan:
        cols_to_keep.append(col)

print(f"  - Después de filtrar NaNs > {threshold_nan*100}%: {len(cols_to_keep)} variables")

# Eliminar features de varianza casi nula
from sklearn.feature_selection import VarianceThreshold
imputer_temp = SimpleImputer(strategy='median')
X_temp = imputer_temp.fit_transform(train_df[cols_to_keep])
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_temp)

cols_to_keep = [col for col, keep in zip(cols_to_keep, selector.get_support()) if keep]
print(f"  - Después de filtrar varianza baja: {len(cols_to_keep)} variables")

# Eliminar features altamente correlacionadas
def eliminar_correlacionadas(df, features, threshold=0.95):
    """Elimina features con correlación muy alta"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(df[features]),
        columns=features
    )
    
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    features_filtered = [f for f in features if f not in to_drop]
    
    return features_filtered, to_drop

cols_to_keep, dropped_corr = eliminar_correlacionadas(train_df, cols_to_keep, threshold=0.95)
print(f"  - Después de eliminar correlacionadas: {len(cols_to_keep)} variables")
print(f"  - Variables eliminadas por correlación: {len(dropped_corr)}")

# Agregar features espaciales y categóricas importantes
feature_cols = cols_to_keep + spatial_cols + ['mean_distance_neighbors', 'min_distance_neighbor']

# Agregar algunas categóricas importantes
important_cats = ['Watershed', 'Streamsize', 'HERlvl1Name']
for cat in important_cats:
    if cat in categorical_cols:
        feature_cols.append(cat)

print(f"\n✓ Features finales seleccionadas: {len(feature_cols)}")

# ============================================
# 4. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================
print("\n🔧 Preparando datos para modelado...")

# Preparar X, y para train
X_train = train_df[feature_cols].copy()
y_train = train_df['IBD_EQR_Status'].copy()

# Preparar X para test
X_test = test_df[feature_cols].copy()

# Encodear target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
print(f"  - Clases: {le.classes_}")

# Imputación de valores faltantes
# Numéricas: mediana
# Categóricas: moda
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Imputador numérico
imputer_num = SimpleImputer(strategy='median')
X_train[numeric_features] = imputer_num.fit_transform(X_train[numeric_features])
X_test[numeric_features] = imputer_num.transform(X_test[numeric_features])

# Imputador categórico y encoding
if categorical_features:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])
    
    # One-hot encoding para categóricas
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    
    # Alinear columnas
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Datos preparados:")
print(f"  - X_train: {X_train_scaled.shape}")
print(f"  - X_test: {X_test_scaled.shape}")

# ============================================
# 5. ENTRENAMIENTO DE MODELOS
# ============================================
print("\n🤖 Entrenando modelos...")

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modelo 1: Random Forest
print("\n  📊 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_scores = cross_val_score(rf_model, X_train_scaled, y_train_encoded, 
                            cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"    ✓ F1-Score CV: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

# Modelo 2: XGBoost
print("\n  📊 XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_scores = cross_val_score(xgb_model, X_train_scaled, y_train_encoded, 
                             cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"    ✓ F1-Score CV: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

# Modelo 3: Gradient Boosting
print("\n  📊 Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

gb_scores = cross_val_score(gb_model, X_train_scaled, y_train_encoded, 
                           cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"    ✓ F1-Score CV: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")

# Seleccionar mejor modelo
scores_dict = {
    'RandomForest': rf_scores.mean(),
    'XGBoost': xgb_scores.mean(),
    'GradientBoosting': gb_scores.mean()
}

best_model_name = max(scores_dict, key=scores_dict.get)
print(f"\n🏆 Mejor modelo: {best_model_name} (F1={scores_dict[best_model_name]:.4f})")

# Entrenar mejor modelo en todos los datos
if best_model_name == 'RandomForest':
    best_model = rf_model
elif best_model_name == 'XGBoost':
    best_model = xgb_model
else:
    best_model = gb_model

print("\n  🔄 Entrenando modelo final...")
best_model.fit(X_train_scaled, y_train_encoded)

# ============================================
# 6. PREDICCIÓN Y EVALUACIÓN
# ============================================
print("\n🎯 Realizando predicciones...")

# Predicciones en train (para validación)
y_train_pred = best_model.predict(X_train_scaled)
print("\n📈 Rendimiento en Train:")
print(classification_report(y_train_encoded, y_train_pred, target_names=le.classes_))

# Predicciones en test
y_test_pred = best_model.predict(X_test_scaled)
y_test_labels = le.inverse_transform(y_test_pred)

print(f"\n✓ Predicciones realizadas: {len(y_test_labels)}")
print(f"  Distribución de predicciones:")
unique, counts = np.unique(y_test_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"    {label}: {count} ({count/len(y_test_labels)*100:.2f}%)")

# ============================================
# 7. ANÁLISIS DE IMPORTANCIA
# ============================================
print("\n📊 Top 20 Features más importantes:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# ============================================
# 8. GUARDAR RESULTADOS
# ============================================
print("\n💾 Guardando resultados...")

# Crear DataFrame de resultados
results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels
})

# Guardar CSV
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"
import os
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, f'predictions_{best_model_name}.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"✓ Archivo guardado: {output_file}")
print(f"  - Registros: {len(results_df)}")

# Guardar también estadísticas del modelo
stats_file = os.path.join(output_path, f'model_stats_{best_model_name}.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write(f"=== ESTADÍSTICAS DEL MODELO ===\n\n")
    f.write(f"Modelo: {best_model_name}\n")
    f.write(f"F1-Score CV: {scores_dict[best_model_name]:.4f}\n\n")
    f.write(f"Features utilizadas: {len(X_train.columns)}\n")
    f.write(f"Muestras de entrenamiento: {len(X_train)}\n")
    f.write(f"Muestras predichas: {len(results_df)}\n\n")
    f.write(f"=== DISTRIBUCIÓN DE PREDICCIONES ===\n")
    for label, count in zip(unique, counts):
        f.write(f"{label}: {count} ({count/len(y_test_labels)*100:.2f}%)\n")
    f.write(f"\n=== TOP 20 FEATURES ===\n")
    f.write(feature_importance.head(20).to_string(index=False))

print(f"✓ Estadísticas guardadas: {stats_file}")

print("\n✨ ¡Proceso completado exitosamente!")
print(f"\n📁 Archivos generados:")
print(f"   1. {output_file}")
print(f"   2. {stats_file}")
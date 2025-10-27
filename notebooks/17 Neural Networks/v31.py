import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================
ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\notebooks\06_cb_regression\completo\tp_full.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"

RANDOM_STATE = 42
N_FOLDS = 10
K_NEIGHBORS = 15

# ============================================
# 1. CARGA DE DATOS
# ============================================
print("=" * 60)
print("SISTEMA DE PREDICCION IBD_EQR_STATUS - VERSION MEJORADA")
print("=" * 60)
print("\n[1/9] Cargando datos...")

df = pd.read_parquet(ruta_archivo)
print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

train_df = df[df['IBD_EQR_Status'].notna()].copy()
test_df = df[df['IBD_EQR_Status'].isna()].copy()

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"\nDistribucion de clases:")
for clase, count in train_df['IBD_EQR_Status'].value_counts().items():
    print(f"  {clase:12s}: {count:6d} ({count/len(train_df)*100:5.2f}%)")

# ============================================
# 2. FEATURE ENGINEERING AVANZADO
# ============================================
print("\n[2/9] Feature Engineering Avanzado...")

def crear_features_espaciales_avanzadas(df_source, df_target, k=15):
    """Features espaciales con agregaciones de vecinos"""
    coords_cols = ['Longitude_Lambert93', 'Latitude_Lambert93']
    valid_mask = df_source[coords_cols].notna().all(axis=1)
    
    if valid_mask.sum() < k:
        return df_target
    
    coords_source = df_source.loc[valid_mask, coords_cols].values
    coords_target = df_target[coords_cols].values
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(coords_source)
    distances, indices = nbrs.kneighbors(coords_target)
    
    # Features de distancia
    df_target['dist_mean'] = distances.mean(axis=1)
    df_target['dist_min'] = distances[:, 0]
    df_target['dist_max'] = distances[:, -1]
    df_target['dist_std'] = distances.std(axis=1)
    df_target['dist_range'] = distances[:, -1] - distances[:, 0]
    
    # Features de densidad
    df_target['neighbor_density'] = k / (np.pi * distances.mean(axis=1)**2 + 1e-10)
    
    # Agregaciones de vecinos si hay target
    if 'IBD_EQR' in df_source.columns:
        valid_source = df_source[valid_mask].reset_index(drop=True)
        
        ibd_neighbors = np.zeros((len(indices), k))
        for i, idx_list in enumerate(indices):
            neighbor_vals = valid_source.iloc[idx_list]['IBD_EQR'].values
            ibd_neighbors[i] = neighbor_vals
        
        df_target['neighbor_ibd_mean'] = np.nanmean(ibd_neighbors, axis=1)
        df_target['neighbor_ibd_std'] = np.nanstd(ibd_neighbors, axis=1)
        df_target['neighbor_ibd_min'] = np.nanmin(ibd_neighbors, axis=1)
        df_target['neighbor_ibd_max'] = np.nanmax(ibd_neighbors, axis=1)
        df_target['neighbor_ibd_range'] = df_target['neighbor_ibd_max'] - df_target['neighbor_ibd_min']
        
        # Percentiles
        df_target['neighbor_ibd_q25'] = np.nanpercentile(ibd_neighbors, 25, axis=1)
        df_target['neighbor_ibd_q75'] = np.nanpercentile(ibd_neighbors, 75, axis=1)
    
    return df_target

def crear_features_biologicas(df):
    """Features de comunidad biológica"""
    species_cols = [col for col in df.columns if len(col) == 7 and col[3:5].isdigit()]
    
    if len(species_cols) > 0:
        species_data = df[species_cols].fillna(0)
        
        # Diversidad (Shannon)
        species_positive = species_data + 1e-10
        proportions = species_positive.div(species_positive.sum(axis=1), axis=0)
        df['shannon_diversity'] = -np.sum(proportions * np.log(proportions + 1e-10), axis=1)
        
        # Riqueza
        df['species_richness'] = (species_data > 0).sum(axis=1)
        
        # Dominancia
        df['max_species_abundance'] = species_data.max(axis=1)
        df['dominance_ratio'] = df['max_species_abundance'] / (species_data.sum(axis=1) + 1e-10)
        
        # Equitabilidad
        df['evenness'] = df['shannon_diversity'] / (np.log(df['species_richness'] + 1) + 1e-10)
        
        # Percentiles
        df['species_abundance_p50'] = species_data.median(axis=1)
        df['species_abundance_p75'] = species_data.quantile(0.75, axis=1)
        df['species_abundance_p90'] = species_data.quantile(0.90, axis=1)
    
    return df

def crear_features_interaccion(df):
    """Interacciones entre features importantes"""
    if 'Altitude' in df.columns:
        if 'Achpy01' in df.columns:
            df['altitude_x_achpy01'] = df['Altitude'] * df['Achpy01'].fillna(0)
        if 'Achmi02' in df.columns:
            df['altitude_x_achmi02'] = df['Altitude'] * df['Achmi02'].fillna(0)
        if 'neighbor_ibd_mean' in df.columns:
            df['altitude_x_neighbor_ibd'] = df['Altitude'] * df['neighbor_ibd_mean'].fillna(0)
    
    if 'Achpy01' in df.columns and 'Achmi02' in df.columns:
        df['ratio_achpy_achmi'] = (df['Achpy01'].fillna(0) + 1) / (df['Achmi02'].fillna(0) + 1)
    
    if 'species_richness' in df.columns and 'neighbor_ibd_mean' in df.columns:
        df['richness_x_neighbor_ibd'] = df['species_richness'] * df['neighbor_ibd_mean'].fillna(0)
    
    return df

# Aplicar feature engineering
print("  Aplicando features espaciales...")
train_df = crear_features_espaciales_avanzadas(train_df, train_df, k=K_NEIGHBORS)
test_df = crear_features_espaciales_avanzadas(train_df, test_df, k=K_NEIGHBORS)

print("  Aplicando features biologicas...")
train_df = crear_features_biologicas(train_df)
test_df = crear_features_biologicas(test_df)

print("  Aplicando features de interaccion...")
train_df = crear_features_interaccion(train_df)
test_df = crear_features_interaccion(test_df)

# ============================================
# 3. SELECCIÓN DE FEATURES MEJORADA
# ============================================
print("\n[3/9] Seleccion de Features...")

id_cols = ['SamplingOperations_code']
target_cols = ['IBD', 'IBD_EQR', 'IBD_EQR_Status']

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in target_cols]

categorical_cols = ['Watershed', 'Streamsize', 'HERlvl1Name', 'CodeDepartement']
categorical_cols = [col for col in categorical_cols if col in df.columns]

# Filtrar por % de NaNs
threshold_nan = 0.70
numeric_cols_filtered = []
for col in numeric_cols:
    nan_ratio = train_df[col].isna().sum() / len(train_df)
    if nan_ratio < threshold_nan:
        numeric_cols_filtered.append(col)

print(f"  Variables numericas validas: {len(numeric_cols_filtered)}")

# Eliminar varianza baja
from sklearn.feature_selection import VarianceThreshold
imputer_temp = SimpleImputer(strategy='median')
X_temp = imputer_temp.fit_transform(train_df[numeric_cols_filtered])
selector = VarianceThreshold(threshold=0.005)
selector.fit(X_temp)
numeric_cols_filtered = [col for col, keep in zip(numeric_cols_filtered, selector.get_support()) if keep]

print(f"  Despues de filtrar varianza: {len(numeric_cols_filtered)}")

# Eliminar correlacionadas
def eliminar_alta_correlacion(df, features, threshold=0.90):
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [f for f in features if f not in to_drop]

numeric_cols_filtered = eliminar_alta_correlacion(train_df, numeric_cols_filtered, 0.90)
print(f"  Despues de eliminar correlacionadas: {len(numeric_cols_filtered)}")

feature_cols = numeric_cols_filtered + categorical_cols
print(f"\nTotal features seleccionadas: {len(feature_cols)}")

# ============================================
# 4. PREPARACIÓN DE DATOS
# ============================================
print("\n[4/9] Preparacion de Datos...")

X_train = train_df[feature_cols].copy()
y_train = train_df['IBD_EQR_Status'].copy()
X_test = test_df[feature_cols].copy()

# Label encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Imputación
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

imputer_num = SimpleImputer(strategy='median')
X_train[numeric_features] = imputer_num.fit_transform(X_train[numeric_features])
X_test[numeric_features] = imputer_num.transform(X_test[numeric_features])

if categorical_features:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])
    
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  X_train: {X_train_scaled.shape}")
print(f"  X_test: {X_test_scaled.shape}")

# ============================================
# 5. BALANCEO MANUAL DE CLASES
# ============================================
print("\n[5/9] Balanceando clases...")

def manual_oversample(X, y, sampling_strategy='auto', random_state=42):
    """
    Oversampling manual usando sklearn.utils.resample
    """
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_resampled = []
    y_resampled = []
    
    for cls in unique:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        # Definir target count
        if sampling_strategy == 'auto':
            target_count = int(max_count * 0.7) if counts[cls] < max_count * 0.5 else len(X_cls)
        else:
            target_count = max_count
        
        if len(X_cls) < target_count:
            X_cls_resampled = resample(X_cls, 
                                      n_samples=target_count, 
                                      random_state=random_state,
                                      replace=True)
            y_cls_resampled = np.full(target_count, cls)
        else:
            X_cls_resampled = X_cls
            y_cls_resampled = y_cls
        
        X_resampled.append(X_cls_resampled)
        y_resampled.append(y_cls_resampled)
    
    return np.vstack(X_resampled), np.hstack(y_resampled)

# Aplicar balanceo
X_train_balanced, y_train_balanced = manual_oversample(
    X_train_scaled, 
    y_train_encoded, 
    sampling_strategy='auto',
    random_state=RANDOM_STATE
)

print(f"  Original: {X_train_scaled.shape[0]} -> Balanceado: {X_train_balanced.shape[0]}")
for cls in range(len(le.classes_)):
    count = (y_train_balanced == cls).sum()
    print(f"  {le.classes_[cls]:12s}: {count:6d}")

# ============================================
# 6. CALCULAR CLASS WEIGHTS
# ============================================
print("\n[6/9] Calculando pesos de clase...")

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)

class_weight_dict = dict(enumerate(class_weights))
print(f"  Pesos calculados: {class_weight_dict}")

# Para XGBoost (scale_pos_weight - solo aplica para binario, usaremos sample_weight)
sample_weights = np.array([class_weight_dict[y] for y in y_train_balanced])

# ============================================
# 7. ENTRENAMIENTO CON REGULARIZACIÓN
# ============================================
print("\n[7/9] Entrenamiento de Modelos con Regularizacion...")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# XGBoost con regularización
print("\n  Modelo 1: XGBoost Regularizado")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_balanced, y_train_balanced)):
    X_tr, X_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
    y_tr, y_val = y_train_balanced[train_idx], y_train_balanced[val_idx]
    w_tr = sample_weights[train_idx]
    
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    y_pred = xgb_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    xgb_scores.append(acc)
    print(f"    Fold {fold+1:2d}: Accuracy = {acc:.4f}")

print(f"  XGBoost CV: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")

# LightGBM con regularización
print("\n  Modelo 2: LightGBM Regularizado")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_samples=20,
    min_split_gain=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

lgb_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_balanced, y_train_balanced)):
    X_tr, X_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
    y_tr, y_val = y_train_balanced[train_idx], y_train_balanced[val_idx]
    
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    y_pred = lgb_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    lgb_scores.append(acc)
    print(f"    Fold {fold+1:2d}: Accuracy = {acc:.4f}")

print(f"  LightGBM CV: {np.mean(lgb_scores):.4f} (+/- {np.std(lgb_scores):.4f})")

# Random Forest regularizado
print("\n  Modelo 3: Random Forest Regularizado")
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    max_samples=0.7,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_balanced, y_train_balanced)):
    X_tr, X_val = X_train_balanced[train_idx], X_train_balanced[val_idx]
    y_tr, y_val = y_train_balanced[train_idx], y_train_balanced[val_idx]
    
    rf_model.fit(X_tr, y_tr)
    y_pred = rf_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    rf_scores.append(acc)
    print(f"    Fold {fold+1:2d}: Accuracy = {acc:.4f}")

print(f"  Random Forest CV: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})")

# ============================================
# 8. ENSEMBLE VOTING
# ============================================
print("\n[8/9] Creando Ensemble Voting...")

# Modelos finales
xgb_final = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=5, gamma=0.2,
    reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1
)

lgb_final = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
    reg_alpha=0.5, reg_lambda=2.0, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)

rf_final = RandomForestClassifier(
    n_estimators=400, max_depth=15, min_samples_split=20, min_samples_leaf=10,
    max_features='sqrt', max_samples=0.7, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_final),
        ('lgb', lgb_final),
        ('rf', rf_final)
    ],
    voting='soft',
    weights=[2, 2, 1],
    n_jobs=-1
)

print("  Entrenando ensemble...")
xgb_final.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights)
lgb_final.fit(X_train_balanced, y_train_balanced)
rf_final.fit(X_train_balanced, y_train_balanced)
voting_model.fit(X_train_balanced, y_train_balanced)

# Evaluar en datos originales
y_train_pred_ensemble = voting_model.predict(X_train_scaled)
train_acc = accuracy_score(y_train_encoded, y_train_pred_ensemble)
train_f1 = f1_score(y_train_encoded, y_train_pred_ensemble, average='weighted')

print(f"\n  Rendimiento en Train (sin oversample):")
print(f"    Accuracy: {train_acc:.4f}")
print(f"    F1-Score: {train_f1:.4f}")

# ============================================
# 9. PREDICCIÓN Y GUARDADO
# ============================================
print("\n[9/9] Generando Predicciones...")

y_test_pred = voting_model.predict(X_test_scaled)
y_test_labels = le.inverse_transform(y_test_pred)

print("\nDistribucion de predicciones:")
unique, counts = np.unique(y_test_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)")

print("\nReporte detallado en Train:")
print(classification_report(y_train_encoded, y_train_pred_ensemble, 
                          target_names=le.classes_, digits=4))

# Guardar
import os
os.makedirs(output_path, exist_ok=True)

results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels
})

output_file = os.path.join(output_path, 'predictions_ensemble_v2.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n  CSV guardado: predictions_ensemble_v2.csv")

# Estadísticas
stats_file = os.path.join(output_path, 'model_stats_ensemble_v2.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ESTADISTICAS DEL MODELO ENSEMBLE V2\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Configuracion:\n")
    f.write(f"  - Folds CV: {N_FOLDS}\n")
    f.write(f"  - K-Neighbors: {K_NEIGHBORS}\n")
    f.write(f"  - Features: {X_train.shape[1]}\n")
    f.write(f"  - Muestras Train: {len(X_train)}\n")
    f.write(f"  - Muestras Balanceadas: {len(X_train_balanced)}\n\n")
    f.write(f"Rendimiento CV:\n")
    f.write(f"  - XGBoost: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})\n")
    f.write(f"  - LightGBM: {np.mean(lgb_scores):.4f} (+/- {np.std(lgb_scores):.4f})\n")
    f.write(f"  - Random Forest: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})\n\n")
    f.write(f"Rendimiento Train Final:\n")
    f.write(f"  - Accuracy: {train_acc:.4f}\n")
    f.write(f"  - F1-Score: {train_f1:.4f}\n\n")
    f.write("Predicciones:\n")
    for label, count in zip(unique, counts):
        f.write(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)\n")

print(f"  Estadisticas guardadas: model_stats_ensemble_v2.txt")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nArchivos generados en: {output_path}")
print(f"  1. predictions_ensemble_v2.csv")
print(f"  2. model_stats_ensemble_v2.txt")
print(f"\nAccuracy en Train: {train_acc:.2%}")
print(f"Predicciones realizadas: {len(results_df)}")
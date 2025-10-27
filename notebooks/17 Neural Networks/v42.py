import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5
K_NEIGHBORS = 20

ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\data\processed\03_CLEAN_COMPLETE_DF_02.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"

np.random.seed(RANDOM_STATE)

def matern_kernel(distances, length_scale, nu=1.5):
    if nu == 1.5:
        sqrt3_d = np.sqrt(3) * distances / (length_scale + 1e-10)
        K = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    elif nu == 2.5:
        sqrt5_d = np.sqrt(5) * distances / (length_scale + 1e-10)
        K = (1 + sqrt5_d + (5/3) * (distances / (length_scale + 1e-10))**2) * np.exp(-sqrt5_d)
    else:
        K = np.exp(-distances / (length_scale + 1e-10))
    return K

def build_covariance_matrix(coords, length_scale, sigma_sq, nugget=1e-4):
    distances = cdist(coords, coords, metric='euclidean')
    K = matern_kernel(distances, length_scale, nu=1.5)
    Sigma = sigma_sq * K + nugget * np.eye(len(coords))
    return Sigma, distances

def kriging_prediction(coords_train, coords_test, residuals_train, length_scale, sigma_sq, nugget=1e-4):
    n_train = coords_train.shape[0]
    n_test = coords_test.shape[0]
    
    Sigma_train, _ = build_covariance_matrix(coords_train, length_scale, sigma_sq, nugget)
    
    dist_cross = cdist(coords_test, coords_train, metric='euclidean')
    K_cross = matern_kernel(dist_cross, length_scale, nu=1.5)
    Sigma_cross = sigma_sq * K_cross
    
    try:
        L, lower = cho_factor(Sigma_train)
        kriging_weights = cho_solve((L, lower), residuals_train)
        kriging_pred = Sigma_cross @ kriging_weights
        
        kriging_var = np.zeros(n_test)
        for i in range(n_test):
            k_star = Sigma_cross[i, :]
            v = cho_solve((L, lower), k_star)
            kriging_var[i] = sigma_sq - np.dot(k_star, v)
        
        return kriging_pred, kriging_var
    except:
        return np.zeros(n_test), np.ones(n_test)

def optimize_gp_parameters(coords, residuals, initial_params=[10.0, 1.0, 0.01]):
    def negative_log_likelihood(params):
        length_scale, sigma_sq, nugget = np.exp(params)
        
        try:
            Sigma, _ = build_covariance_matrix(coords, length_scale, sigma_sq, nugget)
            L, lower = cho_factor(Sigma)
            
            alpha = cho_solve((L, lower), residuals)
            
            nll = 0.5 * np.dot(residuals, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * len(residuals) * np.log(2 * np.pi)
            
            return nll
        except:
            return 1e10
    
    result = minimize(negative_log_likelihood, np.log(initial_params), method='L-BFGS-B')
    
    optimal_params = np.exp(result.x)
    return optimal_params[0], optimal_params[1], optimal_params[2]

def create_spatial_features(coords_train, coords_target, X_train, y_train, k=20):
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(coords_train)
    distances, indices = nbrs.kneighbors(coords_target)
    
    features = {}
    
    features['dist_mean'] = distances.mean(axis=1)
    features['dist_min'] = distances[:, 0]
    features['dist_max'] = distances[:, -1]
    features['dist_std'] = distances.std(axis=1)
    features['dist_range'] = distances[:, -1] - distances[:, 0]
    
    features['neighbor_density'] = k / (np.pi * distances.mean(axis=1)**2 + 1e-10)
    
    if y_train is not None:
        neighbor_labels = np.array([y_train[idx] for idx in indices])
        
        features['neighbor_mode'] = np.array([np.bincount(nl).argmax() for nl in neighbor_labels])
        features['neighbor_entropy'] = np.array([
            -np.sum((np.bincount(nl) / k) * np.log(np.bincount(nl) / k + 1e-10))
            for nl in neighbor_labels
        ])
        
        for i in range(5):
            features[f'neighbor_class_{i}_ratio'] = np.array([(nl == i).sum() / k for nl in neighbor_labels])
    
    for col_idx in range(X_train.shape[1]):
        col_values = X_train[:, col_idx]
        neighbor_values = np.array([[col_values[idx] for idx in idx_list] for idx_list in indices])
        
        if col_idx < 10:
            features[f'neighbor_mean_{col_idx}'] = neighbor_values.mean(axis=1)
            features[f'neighbor_std_{col_idx}'] = neighbor_values.std(axis=1)
    
    return pd.DataFrame(features)

def create_biological_features(df):
    species_cols = [col for col in df.columns if len(col) == 7 and col[3:5].isdigit()]
    
    if len(species_cols) > 0:
        species_data = df[species_cols].fillna(0)
        
        features = {}
        
        species_positive = species_data + 1e-10
        proportions = species_positive.div(species_positive.sum(axis=1), axis=0)
        features['shannon_diversity'] = -np.sum(proportions * np.log(proportions + 1e-10), axis=1)
        features['species_richness'] = (species_data > 0).sum(axis=1)
        features['max_species_abundance'] = species_data.max(axis=1)
        features['dominance_ratio'] = features['max_species_abundance'] / (species_data.sum(axis=1) + 1e-10)
        features['evenness'] = features['shannon_diversity'] / (np.log(features['species_richness'] + 1) + 1e-10)
        features['species_p50'] = species_data.median(axis=1)
        features['species_p75'] = species_data.quantile(0.75, axis=1)
        features['species_p90'] = species_data.quantile(0.90, axis=1)
        
        return pd.DataFrame(features)
    return pd.DataFrame()

print("=" * 60)
print("CATBOOST-GLS: GRADIENT BOOSTING + GAUSSIAN PROCESS KRIGING")
print("=" * 60)

print("\n[1/7] Cargando datos...")
df = pd.read_parquet(ruta_archivo)
train_df = df[df['IBD_EQR_Status'].notna()].copy()
test_df = df[df['IBD_EQR_Status'].isna()].copy()

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

target_cols = ['IBD', 'IBD_EQR', 'IBD_EQR_Status']
spatial_cols = ['Longitude_Lambert93', 'Latitude_Lambert93', 'Altitude']

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in target_cols]

threshold_nan = 0.70
numeric_cols = [col for col in numeric_cols 
                if train_df[col].isna().sum() / len(train_df) < threshold_nan]

print(f"Features numericas base: {len(numeric_cols)}")

X_train = train_df[numeric_cols].copy()
y_train = train_df['IBD_EQR_Status'].copy()
X_test = test_df[numeric_cols].copy()

coords_train = train_df[spatial_cols].values
coords_test = test_df[spatial_cols].values

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

coord_scaler = StandardScaler()
coords_train_scaled = coord_scaler.fit_transform(coords_train)
coords_test_scaled = coord_scaler.transform(coords_test)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

print(f"Clases: {le.classes_}")

print("\n[2/7] Creando features espaciales...")

spatial_train = create_spatial_features(coords_train_scaled, coords_train_scaled, 
                                        X_train_imputed, y_train_encoded, k=K_NEIGHBORS)
spatial_test = create_spatial_features(coords_train_scaled, coords_test_scaled,
                                       X_train_imputed, y_train_encoded, k=K_NEIGHBORS)

print(f"Features espaciales: {spatial_train.shape[1]}")

bio_train = create_biological_features(train_df)
bio_test = create_biological_features(test_df)

if not bio_train.empty:
    print(f"Features biologicas: {bio_train.shape[1]}")

X_train_enhanced = np.hstack([
    X_train_imputed,
    coords_train_scaled,
    spatial_train.values
])

X_test_enhanced = np.hstack([
    X_test_imputed,
    coords_test_scaled,
    spatial_test.values
])

if not bio_train.empty:
    X_train_enhanced = np.hstack([X_train_enhanced, bio_train.values])
    X_test_enhanced = np.hstack([X_test_enhanced, bio_test.values])

print(f"Features totales: {X_train_enhanced.shape[1]}")

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train_enhanced = selector.fit_transform(X_train_enhanced)
X_test_enhanced = selector.transform(X_test_enhanced)

print(f"Despues de filtrar varianza: {X_train_enhanced.shape[1]}")

print("\n[3/7] Entrenamiento CatBoost con CV...")

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []
catboost_models = []
gp_params_list = []
residuals_train_list = []
coords_train_list = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_enhanced, y_train_encoded)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    
    X_tr = X_train_enhanced[train_idx]
    X_val = X_train_enhanced[val_idx]
    y_tr = y_train_encoded[train_idx]
    y_val = y_train_encoded[val_idx]
    coords_tr = coords_train_scaled[train_idx]
    coords_val = coords_train_scaled[val_idx]
    
    cat_model = cb.CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        border_count=128,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )
    
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    y_val_pred = cat_model.predict(X_val).flatten()
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"    CatBoost Accuracy: {val_acc:.4f}")
    
    print(f"    Optimizando parametros GP...")
    y_tr_pred = cat_model.predict(X_tr).flatten()
    residuals_tr = (y_tr - y_tr_pred).astype(float)
    
    length_scale, sigma_sq, nugget = optimize_gp_parameters(coords_tr, residuals_tr)
    
    print(f"    GP params: length_scale={length_scale:.2f}, sigma_sq={sigma_sq:.3f}, nugget={nugget:.5f}")
    
    kriging_correction_val, kriging_var_val = kriging_prediction(
        coords_tr, coords_val, residuals_tr, length_scale, sigma_sq, nugget
    )
    
    y_val_corrected = y_val_pred + kriging_correction_val
    y_val_corrected_int = np.clip(np.round(y_val_corrected), 0, len(le.classes_)-1).astype(int)
    
    val_acc_corrected = accuracy_score(y_val, y_val_corrected_int)
    
    print(f"    Accuracy con kriging: {val_acc_corrected:.4f}")
    print(f"    Mejora: {(val_acc_corrected - val_acc)*100:+.2f}%")
    
    fold_scores.append(val_acc_corrected)
    catboost_models.append(cat_model)
    gp_params_list.append((length_scale, sigma_sq, nugget))
    residuals_train_list.append(residuals_tr)
    coords_train_list.append(coords_tr)

print(f"\nCV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

print("\n[4/7] Entrenamiento modelo final...")

cat_final = cb.CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_strength=0.5,
    bagging_temperature=0.5,
    border_count=128,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=RANDOM_STATE,
    verbose=100
)

cat_final.fit(X_train_enhanced, y_train_encoded)

y_train_pred_final = cat_final.predict(X_train_enhanced).flatten()
residuals_final = (y_train_encoded - y_train_pred_final).astype(float)

length_scale_final, sigma_sq_final, nugget_final = optimize_gp_parameters(
    coords_train_scaled, residuals_final
)

print(f"\nGP params finales: length_scale={length_scale_final:.2f}, sigma_sq={sigma_sq_final:.3f}")

print("\n[5/7] Prediccion en test...")

y_test_pred = cat_final.predict(X_test_enhanced).flatten()

kriging_correction_test, kriging_var_test = kriging_prediction(
    coords_train_scaled, coords_test_scaled, residuals_final,
    length_scale_final, sigma_sq_final, nugget_final
)

y_test_corrected = y_test_pred + kriging_correction_test
y_test_final = np.clip(np.round(y_test_corrected), 0, len(le.classes_)-1).astype(int)

y_test_labels = le.inverse_transform(y_test_final)

confidence_scores = 1.0 / (1.0 + kriging_var_test)

unique, counts = np.unique(y_test_labels, return_counts=True)
print("\nDistribucion de predicciones:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)")

print(f"\nConfianza promedio: {confidence_scores.mean():.3f}")
print(f"Confianza min/max: {confidence_scores.min():.3f} / {confidence_scores.max():.3f}")

print("\n[6/7] Evaluacion en train...")

y_train_pred_corrected = y_train_pred_final + kriging_prediction(
    coords_train_scaled, coords_train_scaled, residuals_final,
    length_scale_final, sigma_sq_final, nugget_final
)[0]

y_train_final = np.clip(np.round(y_train_pred_corrected), 0, len(le.classes_)-1).astype(int)

train_acc = accuracy_score(y_train_encoded, y_train_final)
train_f1 = f1_score(y_train_encoded, y_train_final, average='weighted')

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Train F1-Score: {train_f1:.4f}")

print("\nReporte detallado:")
print(classification_report(y_train_encoded, y_train_final, target_names=le.classes_, digits=4))

print("\n[7/7] Guardando resultados...")

import os
os.makedirs(output_path, exist_ok=True)

results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels,
    'confidence_score': confidence_scores,
    'kriging_correction': kriging_correction_test
})

output_file = os.path.join(output_path, 'predictions_catboost_gls.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"CSV guardado: predictions_catboost_gls.csv")

feature_importance = cat_final.get_feature_importance()
top_features_idx = np.argsort(feature_importance)[-20:][::-1]

stats_file = os.path.join(output_path, 'model_stats_catboost_gls.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("CATBOOST-GLS: GRADIENT BOOSTING + GAUSSIAN PROCESS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Arquitectura:\n")
    f.write("  - CatBoost para funcion media (no lineal)\n")
    f.write("  - Gaussian Process para residuos (covarianza espacial)\n")
    f.write("  - Kriging para prediccion espacial\n\n")
    f.write(f"CatBoost parametros:\n")
    f.write(f"  - Iterations: 800\n")
    f.write(f"  - Depth: 8\n")
    f.write(f"  - Learning rate: 0.03\n")
    f.write(f"  - L2 regularization: 5.0\n\n")
    f.write(f"GP parametros finales:\n")
    f.write(f"  - Length scale: {length_scale_final:.3f}\n")
    f.write(f"  - Sigma^2: {sigma_sq_final:.3f}\n")
    f.write(f"  - Nugget: {nugget_final:.5f}\n\n")
    f.write(f"Features:\n")
    f.write(f"  - Numericas base: {len(numeric_cols)}\n")
    f.write(f"  - Espaciales: {spatial_train.shape[1]}\n")
    if not bio_train.empty:
        f.write(f"  - Biologicas: {bio_train.shape[1]}\n")
    f.write(f"  - Total: {X_train_enhanced.shape[1]}\n\n")
    f.write(f"Rendimiento:\n")
    f.write(f"  - CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})\n")
    f.write(f"  - Train Accuracy: {train_acc:.4f}\n")
    f.write(f"  - Train F1-Score: {train_f1:.4f}\n\n")
    f.write("Top 20 Features:\n")
    for idx in top_features_idx[:20]:
        f.write(f"  Feature {idx}: {feature_importance[idx]:.4f}\n")
    f.write("\nPredicciones Test:\n")
    for label, count in zip(unique, counts):
        f.write(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)\n")

print(f"Estadisticas guardadas: model_stats_catboost_gls.txt")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nArchivos generados:")
print(f"  1. predictions_catboost_gls.csv")
print(f"  2. model_stats_catboost_gls.txt")
print(f"\nCV Accuracy: {np.mean(fold_scores):.2%}")
print(f"Train Accuracy: {train_acc:.2%}")
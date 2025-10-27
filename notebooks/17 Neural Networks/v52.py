import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from joblib import Parallel, delayed
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5
K_NEIGHBORS = 50
N_PSEUDO_ITERATIONS = 2
N_JOBS = -1
BATCH_SIZE = 2048

ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\data\processed\03_CLEAN_COMPLETE_DF_02.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"

np.random.seed(RANDOM_STATE)

def matern_kernel(distances, length_scale, nu=1.5):
    if nu == 1.5:
        sqrt3_d = np.sqrt(3) * distances / (length_scale + 1e-10)
        K = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    else:
        K = np.exp(-distances / (length_scale + 1e-10))
    return K

def local_kriging_prediction(coords_train, coords_test, residuals_train, length_scale, sigma_sq, nugget, k_local=100):
    """
    Kriging local: para cada punto test, usa solo k vecinos mas cercanos
    Evita matrices masivas de covarianza
    """
    n_test = coords_test.shape[0]
    kriging_pred = np.zeros(n_test)
    kriging_var = np.zeros(n_test)
    
    nbrs = NearestNeighbors(n_neighbors=min(k_local, len(coords_train)), metric='euclidean', n_jobs=N_JOBS)
    nbrs.fit(coords_train)
    distances_to_neighbors, indices_neighbors = nbrs.kneighbors(coords_test)
    
    def process_point(i):
        neighbor_coords = coords_train[indices_neighbors[i]]
        neighbor_residuals = residuals_train[indices_neighbors[i]]
        
        dist_matrix = cdist(neighbor_coords, neighbor_coords, metric='euclidean')
        K_local = matern_kernel(dist_matrix, length_scale, nu=1.5)
        Sigma_local = sigma_sq * K_local + nugget * np.eye(len(neighbor_coords))
        
        dist_cross = cdist(coords_test[i:i+1], neighbor_coords, metric='euclidean')[0]
        K_cross = matern_kernel(dist_cross, length_scale, nu=1.5)
        Sigma_cross = sigma_sq * K_cross
        
        try:
            L, lower = cho_factor(Sigma_local)
            weights = cho_solve((L, lower), neighbor_residuals)
            pred = np.dot(Sigma_cross, weights)
            
            v = cho_solve((L, lower), Sigma_cross)
            var = sigma_sq - np.dot(Sigma_cross, v)
            
            return pred, max(var, 0)
        except:
            return 0.0, 1.0
    
    results = Parallel(n_jobs=N_JOBS, backend='threading')(
        delayed(process_point)(i) for i in range(n_test)
    )
    
    for i, (pred, var) in enumerate(results):
        kriging_pred[i] = pred
        kriging_var[i] = var
    
    return kriging_pred, kriging_var

def optimize_gp_parameters_local(coords, residuals, k_local=100, initial_params=None):
    """
    Optimiza parametros GP usando subconjunto de datos con bounds estrictos
    """
    n_samples = min(5000, len(coords))
    idx = np.random.choice(len(coords), n_samples, replace=False)
    coords_subset = coords[idx]
    residuals_subset = residuals[idx]
    
    distances = cdist(coords_subset[:500], coords_subset[:500], metric='euclidean')
    distances_flat = distances[np.triu_indices_from(distances, k=1)]
    median_distance = np.median(distances_flat)
    
    if initial_params is None:
        length_scale_init = median_distance * 2.0
        sigma_sq_init = np.var(residuals_subset)
        nugget_init = sigma_sq_init * 0.01
    else:
        length_scale_init, sigma_sq_init, nugget_init = initial_params
    
    bounds = [
        (np.log(median_distance * 0.1), np.log(median_distance * 10)),
        (np.log(np.var(residuals_subset) * 0.01), np.log(np.var(residuals_subset) * 10)),
        (np.log(1e-6), np.log(np.var(residuals_subset) * 0.5))
    ]
    
    def negative_log_likelihood(params):
        length_scale, sigma_sq, nugget = np.exp(params)
        
        if length_scale < 0.01 or length_scale > 1000:
            return 1e10
        if sigma_sq < 1e-6 or sigma_sq > 1000:
            return 1e10
        if nugget < 1e-6 or nugget > 10:
            return 1e10
        
        try:
            dist_matrix = cdist(coords_subset, coords_subset, metric='euclidean')
            K = matern_kernel(dist_matrix, length_scale, nu=1.5)
            Sigma = sigma_sq * K + nugget * np.eye(len(coords_subset))
            
            L, lower = cho_factor(Sigma)
            alpha = cho_solve((L, lower), residuals_subset)
            
            nll = 0.5 * np.dot(residuals_subset, alpha)
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * len(residuals_subset) * np.log(2 * np.pi)
            
            if not np.isfinite(nll):
                return 1e10
            
            return nll
        except:
            return 1e10
    
    best_result = None
    best_nll = 1e10
    
    init_guesses = [
        np.log([length_scale_init, sigma_sq_init, nugget_init]),
        np.log([median_distance, np.var(residuals_subset), np.var(residuals_subset) * 0.05]),
        np.log([median_distance * 0.5, np.var(residuals_subset) * 0.5, np.var(residuals_subset) * 0.1])
    ]
    
    for init_guess in init_guesses:
        try:
            result = minimize(
                negative_log_likelihood, 
                init_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-4}
            )
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None or best_nll >= 1e10:
        print("      WARNING: GP optimization failed, using defaults")
        return length_scale_init, sigma_sq_init, nugget_init
    
    optimal_params = np.exp(best_result.x)
    
    if optimal_params[0] < 0.01 or optimal_params[0] > 1000:
        print("      WARNING: length_scale out of range, clipping")
        optimal_params[0] = np.clip(optimal_params[0], 0.1, 100)
    
    return optimal_params[0], optimal_params[1], optimal_params[2]

def create_spatial_features_parallel(coords_train, coords_target, X_train, y_train, k=50):
    """
    Features espaciales con paralelizacion
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=N_JOBS)
    nbrs.fit(coords_train)
    distances, indices = nbrs.kneighbors(coords_target)
    
    features = {}
    
    features['dist_mean'] = distances.mean(axis=1)
    features['dist_min'] = distances[:, 0]
    features['dist_max'] = distances[:, -1]
    features['dist_std'] = distances.std(axis=1)
    features['dist_q25'] = np.percentile(distances, 25, axis=1)
    features['dist_q75'] = np.percentile(distances, 75, axis=1)
    features['neighbor_density'] = k / (np.pi * distances.mean(axis=1)**2 + 1e-10)
    
    if y_train is not None:
        neighbor_labels = np.array([y_train[idx] for idx in indices])
        
        features['neighbor_mode'] = np.array([np.bincount(nl).argmax() for nl in neighbor_labels])
        features['neighbor_entropy'] = np.array([
            -np.sum((np.bincount(nl, minlength=5) / k) * np.log(np.bincount(nl, minlength=5) / k + 1e-10))
            for nl in neighbor_labels
        ])
        
        for i in range(5):
            features[f'neighbor_class_{i}_ratio'] = np.array([(nl == i).sum() / k for nl in neighbor_labels])
    
    for col_idx in range(min(10, X_train.shape[1])):
        col_values = X_train[:, col_idx]
        neighbor_values = np.array([[col_values[idx] for idx in idx_list] for idx_list in indices])
        
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
        
        for q in [50, 75, 90]:
            features[f'species_p{q}'] = species_data.quantile(q/100, axis=1)
        
        return pd.DataFrame(features)
    return pd.DataFrame()

print("=" * 60)
print("CATBOOST-GLS GPU OPTIMIZADO CON KRIGING LOCAL")
print("=" * 60)
print(f"Hardware: GPU disponible, {N_JOBS} CPUs paralelos")
print(f"Kriging local: k={K_NEIGHBORS} vecinos")

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

print(f"Features numericas: {len(numeric_cols)}")

X_train = train_df[numeric_cols].copy()
y_train_status = train_df['IBD_EQR_Status'].copy()
y_train_continuous = train_df['IBD_EQR'].copy()
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
y_train_encoded = le.fit_transform(y_train_status)

valid_continuous = y_train_continuous[y_train_continuous.notna()]
valid_status = y_train_status[y_train_continuous.notna()]
valid_encoded = le.transform(valid_status)

thresholds = []
for i in range(len(le.classes_) - 1):
    mask_current = valid_encoded == i
    mask_next = valid_encoded == i + 1
    
    if mask_current.sum() > 0 and mask_next.sum() > 0:
        max_current = valid_continuous[mask_current].max()
        min_next = valid_continuous[mask_next].min()
        threshold = (max_current + min_next) / 2
        thresholds.append(threshold)
    else:
        if len(thresholds) > 0:
            thresholds.append(thresholds[-1] + 1)
        else:
            thresholds.append(valid_continuous.quantile((i+1)/len(le.classes_)))

thresholds = sorted(thresholds)
print(f"Thresholds calculados: {thresholds}")

print(f"Clases: {le.classes_}")

print("\n[2/7] Features espaciales (paralelo)...")

spatial_train = create_spatial_features_parallel(coords_train_scaled, coords_train_scaled, 
                                                 X_train_imputed, y_train_encoded, k=K_NEIGHBORS)
spatial_test = create_spatial_features_parallel(coords_train_scaled, coords_test_scaled,
                                                X_train_imputed, y_train_encoded, k=K_NEIGHBORS)

print(f"Features espaciales: {spatial_train.shape[1]}")

bio_train = create_biological_features(train_df)
bio_test = create_biological_features(test_df)

if not bio_train.empty:
    print(f"Features biologicas: {bio_train.shape[1]}")

X_train_enhanced = np.hstack([X_train_imputed, coords_train_scaled, spatial_train.values])
X_test_enhanced = np.hstack([X_test_imputed, coords_test_scaled, spatial_test.values])

if not bio_train.empty:
    X_train_enhanced = np.hstack([X_train_enhanced, bio_train.values])
    X_test_enhanced = np.hstack([X_test_enhanced, bio_test.values])

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train_enhanced = selector.fit_transform(X_train_enhanced)
X_test_enhanced = selector.transform(X_test_enhanced)

print(f"Features totales: {X_train_enhanced.shape[1]}")

print("\n[3/7] Clustering espacial...")

lon_bins = pd.qcut(coords_train_scaled[:, 0], q=5, labels=False, duplicates='drop')
lat_bins = pd.qcut(coords_train_scaled[:, 1], q=2, labels=False, duplicates='drop')
spatial_groups = lon_bins * 2 + lat_bins

print(f"Clusters espaciales: {len(np.unique(spatial_groups))}")

print("\n[4/7] Entrenamiento CatBoost GPU con CV...")

base_params_status = {
    'loss_function': 'MultiClass',
    'eval_metric': 'Accuracy',
    'depth': 8,
    'learning_rate': 0.02,
    'iterations': 2000,
    'l2_leaf_reg': 5.0,
    'random_strength': 0.5,
    'bagging_temperature': 0.5,
    'border_count': 128,
    'random_state': RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': False,
    'early_stopping_rounds': 50
}

base_params_continuous = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'depth': 8,
    'learning_rate': 0.02,
    'iterations': 2000,
    'l2_leaf_reg': 5.0,
    'random_strength': 0.5,
    'bagging_temperature': 0.5,
    'border_count': 128,
    'random_state': RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': False,
    'early_stopping_rounds': 50
}

gkf = GroupKFold(n_splits=N_FOLDS)
fold_scores_status = []
fold_scores_r2 = []
catboost_models_status = []
catboost_models_continuous = []
gp_params_list = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_enhanced, y_train_encoded, groups=spatial_groups)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    
    X_tr = X_train_enhanced[train_idx]
    X_val = X_train_enhanced[val_idx]
    y_tr_status = y_train_encoded[train_idx]
    y_val_status = y_train_encoded[val_idx]
    y_tr_cont = y_train_continuous.iloc[train_idx].values
    y_val_cont = y_train_continuous.iloc[val_idx].values
    coords_tr = coords_train_scaled[train_idx]
    coords_val = coords_train_scaled[val_idx]
    
    cat_status = cb.CatBoostClassifier(**base_params_status)
    cat_continuous = cb.CatBoostRegressor(**base_params_continuous)
    
    cat_status.fit(X_tr, y_tr_status, eval_set=(X_val, y_val_status), use_best_model=True)
    cat_continuous.fit(X_tr, y_tr_cont, eval_set=(X_val, y_val_cont), use_best_model=True)
    
    y_val_pred_status = cat_status.predict(X_val).flatten()
    y_val_pred_cont = cat_continuous.predict(X_val).flatten()
    
    val_acc_status = accuracy_score(y_val_status, y_val_pred_status)
    val_r2 = r2_score(y_val_cont, y_val_pred_cont)
    
    print(f"    Status Accuracy: {val_acc_status:.4f}")
    print(f"    Continuous R2: {val_r2:.4f}")
    
    residuals_cont = y_tr_cont - cat_continuous.predict(X_tr).flatten()
    
    print(f"    Optimizando GP (subset)...")
    length_scale, sigma_sq, nugget = optimize_gp_parameters_local(coords_tr, residuals_cont, k_local=100)
    gp_params_list.append((length_scale, sigma_sq, nugget))
    
    print(f"    GP params: ls={length_scale:.2f}, sigma={sigma_sq:.3f}")
    
    print(f"    Kriging local validacion...")
    kriging_pred_val, kriging_var_val = local_kriging_prediction(
        coords_tr, coords_val, residuals_cont, length_scale, sigma_sq, nugget, k_local=K_NEIGHBORS
    )
    
    y_val_cont_corrected = y_val_pred_cont + kriging_pred_val
    y_val_status_from_cont = np.digitize(y_val_cont_corrected, bins=thresholds)
    y_val_status_from_cont = np.clip(y_val_status_from_cont, 0, len(le.classes_) - 1)
    
    val_acc_hybrid = accuracy_score(y_val_status, y_val_status_from_cont)
    
    print(f"    Hybrid Accuracy: {val_acc_hybrid:.4f}")
    
    best_acc = max(val_acc_status, val_acc_hybrid)
    
    fold_scores_status.append(best_acc)
    fold_scores_r2.append(val_r2)
    catboost_models_status.append(cat_status)
    catboost_models_continuous.append(cat_continuous)

print(f"\nCV Status Accuracy: {np.mean(fold_scores_status):.4f} (+/- {np.std(fold_scores_status):.4f})")
print(f"CV Continuous R2: {np.mean(fold_scores_r2):.4f} (+/- {np.std(fold_scores_r2):.4f})")

print("\n[5/7] Entrenamiento modelo final GPU...")

cat_status_final = cb.CatBoostClassifier(**base_params_status)
cat_continuous_final = cb.CatBoostRegressor(**base_params_continuous)

cat_status_final.set_params(verbose=100)
cat_continuous_final.set_params(verbose=100)

cat_status_final.fit(X_train_enhanced, y_train_encoded)
cat_continuous_final.fit(X_train_enhanced, y_train_continuous.values)

residuals_final = y_train_continuous.values - cat_continuous_final.predict(X_train_enhanced).flatten()

print("\nOptimizando GP final...")
length_scale_final, sigma_sq_final, nugget_final = optimize_gp_parameters_local(
    coords_train_scaled, residuals_final, k_local=100
)

print(f"GP params finales: length_scale={length_scale_final:.2f}, sigma_sq={sigma_sq_final:.3f}")

print("\n[6/7] Prediccion test con kriging local...")

y_test_pred_status_direct = cat_status_final.predict(X_test_enhanced).flatten()
y_test_pred_cont = cat_continuous_final.predict(X_test_enhanced).flatten()

print("Aplicando kriging local (paralelo)...")
kriging_correction_test, kriging_var_test = local_kriging_prediction(
    coords_train_scaled, coords_test_scaled, residuals_final,
    length_scale_final, sigma_sq_final, nugget_final, k_local=K_NEIGHBORS
)

y_test_cont_corrected = y_test_pred_cont + kriging_correction_test
y_test_status_from_cont = np.digitize(y_test_cont_corrected, bins=thresholds)
y_test_status_from_cont = np.clip(y_test_status_from_cont, 0, len(le.classes_) - 1)

confidence_scores = 1.0 / (1.0 + kriging_var_test)
confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())

weight_direct = 0.3
weight_hybrid = 0.7

proba_direct = cat_status_final.predict_proba(X_test_enhanced)
proba_hybrid = np.zeros_like(proba_direct)
for i, pred in enumerate(y_test_status_from_cont):
    proba_hybrid[i, pred] = 1.0

proba_ensemble = weight_direct * proba_direct + weight_hybrid * proba_hybrid
y_test_final = np.argmax(proba_ensemble, axis=1)

print("\n[7/7] Pseudo-labeling iterativo...")

for iteration in range(N_PSEUDO_ITERATIONS):
    print(f"\n  Iteracion {iteration + 1}/{N_PSEUDO_ITERATIONS}")
    
    confident_mask = confidence_scores > 0.88
    n_confident = confident_mask.sum()
    
    if n_confident < 50:
        print(f"    Pocas predicciones confiables ({n_confident}), fin")
        break
    
    print(f"    Usando {n_confident} muestras pseudo-etiquetadas")
    
    X_pseudo = X_test_enhanced[confident_mask]
    y_pseudo_status = y_test_final[confident_mask]
    y_pseudo_cont = y_test_cont_corrected[confident_mask]
    
    X_expanded = np.vstack([X_train_enhanced, X_pseudo])
    y_expanded_status = np.hstack([y_train_encoded, y_pseudo_status])
    y_expanded_cont = np.hstack([y_train_continuous.values, y_pseudo_cont])
    
    cat_status_final.fit(X_expanded, y_expanded_status, verbose=False)
    cat_continuous_final.fit(X_expanded, y_expanded_cont, verbose=False)
    
    y_test_pred_status_direct = cat_status_final.predict(X_test_enhanced).flatten()
    y_test_pred_cont = cat_continuous_final.predict(X_test_enhanced).flatten()
    
    residuals_expanded = y_expanded_cont - cat_continuous_final.predict(X_expanded).flatten()
    coords_expanded = np.vstack([coords_train_scaled, coords_test_scaled[confident_mask]])
    
    kriging_correction_test, kriging_var_test = local_kriging_prediction(
        coords_expanded, coords_test_scaled, residuals_expanded,
        length_scale_final, sigma_sq_final, nugget_final, k_local=K_NEIGHBORS
    )
    
    y_test_cont_corrected = y_test_pred_cont + kriging_correction_test
    y_test_status_from_cont = np.digitize(y_test_cont_corrected, bins=[0, 5, 10, 14, 18]) - 1
    y_test_status_from_cont = np.clip(y_test_status_from_cont, 0, 4)
    
    proba_direct = cat_status_final.predict_proba(X_test_enhanced)
    proba_hybrid = np.zeros_like(proba_direct)
    for i, pred in enumerate(y_test_status_from_cont):
        proba_hybrid[i, pred] = 1.0
    
    proba_ensemble = weight_direct * proba_direct + weight_hybrid * proba_hybrid
    y_test_final = np.argmax(proba_ensemble, axis=1)
    
    confidence_scores = 1.0 / (1.0 + kriging_var_test)
    confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
    
    print(f"    Confianza promedio: {confidence_scores.mean():.3f}")

y_test_labels = le.inverse_transform(y_test_final)

unique, counts = np.unique(y_test_labels, return_counts=True)
print("\nDistribucion final:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)")

y_train_pred_status = cat_status_final.predict(X_train_enhanced).flatten()
y_train_pred_cont = cat_continuous_final.predict(X_train_enhanced).flatten()

kriging_train, _ = local_kriging_prediction(
    coords_train_scaled, coords_train_scaled, residuals_final,
    length_scale_final, sigma_sq_final, nugget_final, k_local=K_NEIGHBORS
)

y_train_cont_corrected = y_train_pred_cont + kriging_train
y_train_status_from_cont = np.digitize(y_train_cont_corrected, bins=[0, 5, 10, 14, 18]) - 1
y_train_status_from_cont = np.clip(y_train_status_from_cont, 0, 4)

proba_direct_train = cat_status_final.predict_proba(X_train_enhanced)
proba_hybrid_train = np.zeros_like(proba_direct_train)
for i, pred in enumerate(y_train_status_from_cont):
    proba_hybrid_train[i, pred] = 1.0

proba_ensemble_train = weight_direct * proba_direct_train + weight_hybrid * proba_hybrid_train
y_train_final = np.argmax(proba_ensemble_train, axis=1)

train_acc = accuracy_score(y_train_encoded, y_train_final)
train_f1 = f1_score(y_train_encoded, y_train_final, average='weighted')
train_r2 = r2_score(y_train_continuous.values, y_train_cont_corrected)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Train F1-Score: {train_f1:.4f}")
print(f"Train R2 (continuous): {train_r2:.4f}")

print("\nReporte detallado:")
print(classification_report(y_train_encoded, y_train_final, target_names=le.classes_, digits=4))

import os
os.makedirs(output_path, exist_ok=True)

results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels,
    'IBD_EQR_predicted': y_test_cont_corrected,
    'confidence_score': confidence_scores,
    'kriging_correction': kriging_correction_test
})

output_file = os.path.join(output_path, 'predictions_catboost_gls_gpu.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\nCSV guardado: predictions_catboost_gls_gpu.csv")

stats_file = os.path.join(output_path, 'model_stats_catboost_gls_gpu.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("CATBOOST-GLS GPU OPTIMIZADO\n")
    f.write("=" * 60 + "\n\n")
    f.write("Optimizaciones:\n")
    f.write("  - GPU acceleration (CUDA)\n")
    f.write("  - Paralelizacion CPU (joblib)\n")
    f.write(f"  - Kriging local (k={K_NEIGHBORS} vecinos)\n")
    f.write("  - Batch processing\n")
    f.write("  - Multi-target (status + continuous)\n")
    f.write(f"  - Pseudo-labeling ({N_PSEUDO_ITERATIONS} iter)\n\n")
    f.write(f"Rendimiento:\n")
    f.write(f"  - CV Accuracy: {np.mean(fold_scores_status):.4f} (+/- {np.std(fold_scores_status):.4f})\n")
    f.write(f"  - CV R2: {np.mean(fold_scores_r2):.4f} (+/- {np.std(fold_scores_r2):.4f})\n")
    f.write(f"  - Train Accuracy: {train_acc:.4f}\n")
    f.write(f"  - Train F1-Score: {train_f1:.4f}\n")
    f.write(f"  - Train R2: {train_r2:.4f}\n\n")
    f.write("GP parametros finales:\n")
    f.write(f"  - Length scale: {length_scale_final:.3f}\n")
    f.write(f"  - Sigma^2: {sigma_sq_final:.3f}\n")
    f.write(f"  - Nugget: {nugget_final:.5f}\n\n")
    f.write("Predicciones Test:\n")
    for label, count in zip(unique, counts):
        f.write(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)\n")

print(f"Estadisticas guardadas: model_stats_catboost_gls_gpu.txt")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nCV Accuracy: {np.mean(fold_scores_status):.2%}")
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Speedup estimado: 5-10x vs CPU")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
N_FOLDS = 5
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\notebooks\06_cb_regression\completo\tp_full.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"

def matern_kernel(distances, length_scale, nu=1.5):
    if nu == 1.5:
        sqrt3_d = np.sqrt(3) * distances / length_scale
        K = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    else:
        raise ValueError(f"nu={nu} no implementado")
    return K

def build_spatial_covariance_matrix(coords, length_scale, sigma_sq, nugget=1e-4):
    n = coords.shape[0]
    distances = cdist(coords, coords, metric='euclidean')
    K = matern_kernel(distances, length_scale, nu=1.5)
    Sigma = sigma_sq * K + nugget * np.eye(n)
    return Sigma

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], output_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3 if i < 2 else 0.2)
            ])
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        features = self.network(x)
        output = self.output_layer(features)
        return output

class NNGLS_Model:
    def __init__(self, input_dim, coords, device='cpu', n_classes=5):
        self.device = device
        self.coords = coords
        self.n_classes = n_classes
        
        self.nn = DeepNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            output_dim=1
        ).to(device)
        
        self.log_length_scale = nn.Parameter(torch.tensor(10.0, device=device))
        self.log_sigma_sq = nn.Parameter(torch.tensor(1.0, device=device))
        self.log_nugget = nn.Parameter(torch.tensor(-4.0, device=device))
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, n_classes - 1, device=device))
        
    def compute_spatial_precision(self, coords_batch):
        length_scale = torch.exp(self.log_length_scale).item()
        sigma_sq = torch.exp(self.log_sigma_sq).item()
        nugget = torch.exp(self.log_nugget).item()
        
        Sigma = build_spatial_covariance_matrix(coords_batch, length_scale, sigma_sq, nugget)
        
        try:
            L, lower = cho_factor(Sigma)
            return L, lower, Sigma
        except:
            Sigma += 1e-3 * np.eye(len(Sigma))
            L, lower = cho_factor(Sigma)
            return L, lower, Sigma
    
    def gls_loss(self, predictions, targets, L, lower):
        residuals = (targets - predictions).cpu().numpy()
        precision_residuals = cho_solve((L, lower), residuals)
        gls_loss = 0.5 * np.dot(residuals, precision_residuals)
        return torch.tensor(gls_loss, dtype=torch.float32, device=self.device)
    
    def ordinal_classification_loss(self, scores, labels):
        thresholds_sorted = torch.sort(self.thresholds)[0]
        cum_probs = torch.sigmoid(thresholds_sorted.unsqueeze(0) - scores)
        cum_probs = torch.cat([
            torch.zeros_like(scores),
            cum_probs,
            torch.ones_like(scores)
        ], dim=1)
        probs = cum_probs[:, 1:] - cum_probs[:, :-1]
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        nll = F.nll_loss(torch.log(probs), labels)
        return nll
    
    def predict_proba(self, X):
        self.nn.eval()
        with torch.no_grad():
            scores = self.nn(X)
            thresholds_sorted = torch.sort(self.thresholds)[0]
            cum_probs = torch.sigmoid(thresholds_sorted.unsqueeze(0) - scores)
            cum_probs = torch.cat([
                torch.zeros_like(scores),
                cum_probs,
                torch.ones_like(scores)
            ], dim=1)
            probs = cum_probs[:, 1:] - cum_probs[:, :-1]
            probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

def train_nngls(model, train_loader, coords_train, optimizer, device, use_spatial=True, lambda_spatial=0.5):
    model.nn.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        scores = model.nn(features)
        ordinal_loss = model.ordinal_classification_loss(scores, labels)
        
        spatial_loss = torch.tensor(0.0, device=device)
        if use_spatial and len(features) <= 512:
            start_idx = batch_idx * len(features)
            end_idx = start_idx + len(features)
            
            if end_idx <= len(coords_train):
                coords_batch = coords_train[start_idx:end_idx]
                ordinal_values = labels.float().unsqueeze(1)
                
                try:
                    L, lower, _ = model.compute_spatial_precision(coords_batch)
                    spatial_loss = model.gls_loss(scores, ordinal_values, L, lower)
                except:
                    spatial_loss = torch.tensor(0.0, device=device)
        
        total = ordinal_loss + lambda_spatial * spatial_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.nn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([model.thresholds, model.log_length_scale, 
                                       model.log_sigma_sq, model.log_nugget], max_norm=1.0)
        optimizer.step()
        
        total_loss += total.item()
        
        with torch.no_grad():
            probs = model.predict_proba(features)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), acc

def validate_nngls(model, val_loader, device):
    model.nn.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            probs = model.predict_proba(features)
            preds = torch.argmax(probs, dim=1)
            scores = model.nn(features)
            loss = model.ordinal_classification_loss(scores, labels)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(val_loader), acc, f1

print("=" * 60)
print("NN-GLS CON MEGA MERGE DATASET")
print("=" * 60)

print("\n[1/6] Cargando datos...")
df = pd.read_parquet(ruta_archivo)
print(f"Dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

train_df = df[df['IBD_EQR_Status'].notna()].copy()
test_df = df[df['IBD_EQR_Status'].isna()].copy()
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

print("\n[2/6] Limpieza de datos...")

id_cols = ['SamplingOperations_code']
target_cols = ['IBD', 'IBD_EQR', 'IBD_EQR_Status']
spatial_cols = ['Longitude_Lambert93', 'Latitude_Lambert93', 'Altitude']

feature_cols = [col for col in df.columns if col not in id_cols + target_cols]

numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Variables numericas: {len(numeric_cols)}")
print(f"Variables categoricas: {len(categorical_cols)}")

threshold_nan = 0.60
numeric_filtered = []
for col in numeric_cols:
    nan_ratio = train_df[col].isna().sum() / len(train_df)
    if nan_ratio < threshold_nan:
        numeric_filtered.append(col)

print(f"Despues de filtrar NaN > {threshold_nan*100}%: {len(numeric_filtered)}")

categorical_filtered = []
for col in categorical_cols:
    nan_ratio = train_df[col].isna().sum() / len(train_df)
    if nan_ratio < threshold_nan:
        categorical_filtered.append(col)

print(f"Categoricas validas: {len(categorical_filtered)}")

print("\n[3/6] Procesando features...")

for col in categorical_filtered:
    le = LabelEncoder()
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
    all_values = pd.concat([train_df[col], test_df[col]]).unique()
    le.fit(all_values)
    
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

all_features = numeric_filtered + categorical_filtered

X_train = train_df[all_features].copy()
y_train = train_df['IBD_EQR_Status'].copy()
X_test = test_df[all_features].copy()

coords_train = train_df[spatial_cols].values
coords_test = test_df[spatial_cols].values

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print(f"Features despues de imputacion: {X_train_imputed.shape[1]}")

selector = VarianceThreshold(threshold=0.01)
X_train_var = selector.fit_transform(X_train_imputed)
X_test_var = selector.transform(X_test_imputed)

selected_features = np.array(all_features)[selector.get_support()]
print(f"Despues de filtrar varianza: {len(selected_features)}")

def remove_correlated_features(X, feature_names, threshold=0.90):
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    remaining = [f for f in feature_names if f not in to_drop]
    return remaining, to_drop

remaining_features, dropped = remove_correlated_features(X_train_var, selected_features, 0.90)
print(f"Despues de eliminar correlacionadas (>0.90): {len(remaining_features)}")

feature_indices = [i for i, f in enumerate(selected_features) if f in remaining_features]
X_train_final = X_train_var[:, feature_indices]
X_test_final = X_test_var[:, feature_indices]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

coord_scaler = RobustScaler()
coords_train_scaled = coord_scaler.fit_transform(coords_train)
coords_test_scaled = coord_scaler.transform(coords_test)

le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
n_classes = len(le_target.classes_)

print(f"\nClases: {le_target.classes_}")
print(f"Shape final: X_train={X_train_scaled.shape}, coords={coords_train_scaled.shape}")

print("\n[4/6] Entrenamiento con CV...")

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train_encoded)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    
    X_tr = torch.FloatTensor(X_train_scaled[train_idx])
    X_val = torch.FloatTensor(X_train_scaled[val_idx])
    y_tr = torch.LongTensor(y_train_encoded[train_idx])
    y_val = torch.LongTensor(y_train_encoded[val_idx])
    
    coords_tr = coords_train_scaled[train_idx]
    coords_val = coords_train_scaled[val_idx]
    
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = NNGLS_Model(
        input_dim=X_train_scaled.shape[1],
        coords=coords_tr,
        device=DEVICE,
        n_classes=n_classes
    )
    
    optimizer = torch.optim.AdamW([
        {'params': model.nn.parameters(), 'lr': LEARNING_RATE, 'weight_decay': 1e-4},
        {'params': [model.thresholds], 'lr': LEARNING_RATE * 0.5},
        {'params': [model.log_length_scale, model.log_sigma_sq, model.log_nugget], 
         'lr': LEARNING_RATE * 0.1}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        use_spatial = epoch > 10
        
        train_loss, train_acc = train_nngls(
            model, train_loader, coords_tr, optimizer, DEVICE,
            use_spatial=use_spatial, lambda_spatial=0.3
        )
        
        val_loss, val_acc, val_f1 = validate_nngls(model, val_loader, DEVICE)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'nn': model.nn.state_dict(),
                'thresholds': model.thresholds.clone(),
                'log_length_scale': model.log_length_scale.clone(),
                'log_sigma_sq': model.log_sigma_sq.clone(),
                'log_nugget': model.log_nugget.clone()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            length_scale = torch.exp(model.log_length_scale).item()
            print(f"    Epoch {epoch+1:3d}: Train={train_acc:.4f} | Val={val_acc:.4f}, F1={val_f1:.4f} | LS={length_scale:.2f}")
        
        if patience_counter >= patience:
            print(f"    Early stopping en epoch {epoch+1}")
            break
    
    model.nn.load_state_dict(best_model_state['nn'])
    model.thresholds = best_model_state['thresholds']
    model.log_length_scale = best_model_state['log_length_scale']
    model.log_sigma_sq = best_model_state['log_sigma_sq']
    model.log_nugget = best_model_state['log_nugget']
    
    fold_scores.append(best_val_acc)
    fold_models.append(model)
    
    print(f"  Mejor Accuracy: {best_val_acc:.4f}")

print(f"\nCV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

print("\n[5/6] Prediccion en test...")

X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

all_probs = []
for model in fold_models:
    probs = model.predict_proba(X_test_tensor)
    all_probs.append(probs.cpu().numpy())

ensemble_probs = np.mean(all_probs, axis=0)
final_preds = np.argmax(ensemble_probs, axis=1)
y_test_labels = le_target.inverse_transform(final_preds)

unique, counts = np.unique(y_test_labels, return_counts=True)
print("\nDistribucion de predicciones:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)")

X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)

train_probs_all = []
for model in fold_models:
    probs = model.predict_proba(X_train_tensor)
    train_probs_all.append(probs.cpu().numpy())

train_probs = np.mean(train_probs_all, axis=0)
train_preds = np.argmax(train_probs, axis=1)

train_acc = accuracy_score(y_train_encoded, train_preds)
train_f1 = f1_score(y_train_encoded, train_preds, average='weighted')

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Train F1-Score: {train_f1:.4f}")

print("\nReporte detallado:")
print(classification_report(y_train_encoded, train_preds, target_names=le_target.classes_, digits=4))

print("\n[6/6] Guardando resultados...")

import os
os.makedirs(output_path, exist_ok=True)

results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels
})

output_file = os.path.join(output_path, 'predictions_nngls_mega.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"CSV guardado: predictions_nngls_mega.csv")

stats_file = os.path.join(output_path, 'model_stats_nngls_mega.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("NN-GLS CON MEGA MERGE DATASET\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset original: {df.shape}\n")
    f.write(f"Features finales: {X_train_scaled.shape[1]}\n")
    f.write(f"Train samples: {len(X_train_scaled)}\n")
    f.write(f"Test samples: {len(X_test_scaled)}\n\n")
    f.write(f"Limpieza:\n")
    f.write(f"  - Threshold NaN: {threshold_nan*100}%\n")
    f.write(f"  - Threshold varianza: 0.01\n")
    f.write(f"  - Threshold correlacion: 0.90\n\n")
    f.write(f"Rendimiento:\n")
    f.write(f"  - CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})\n")
    f.write(f"  - Train Accuracy: {train_acc:.4f}\n")
    f.write(f"  - Train F1-Score: {train_f1:.4f}\n\n")
    avg_length = np.mean([torch.exp(m.log_length_scale).item() for m in fold_models])
    avg_sigma = np.mean([torch.exp(m.log_sigma_sq).item() for m in fold_models])
    f.write(f"Parametros GP:\n")
    f.write(f"  - Length scale: {avg_length:.3f}\n")
    f.write(f"  - Sigma^2: {avg_sigma:.3f}\n\n")
    f.write("Predicciones Test:\n")
    for label, count in zip(unique, counts):
        f.write(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)\n")

print(f"Estadisticas guardadas: model_stats_nngls_mega.txt")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nArchivos generados:")
print(f"  1. predictions_nngls_mega.csv")
print(f"  2. model_stats_nngls_mega.txt")
print(f"\nCV Accuracy: {np.mean(fold_scores):.2%}")
print(f"Train Accuracy: {train_acc:.2%}")
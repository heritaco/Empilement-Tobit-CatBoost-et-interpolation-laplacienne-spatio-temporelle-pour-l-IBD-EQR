import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")

RANDOM_STATE = 42
N_FOLDS = 5
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\notebooks\06_cb_regression\completo\tp_full.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"

# ============================================
# 1. FUNCIONES DE KERNEL ESPACIAL
# ============================================
def matern_kernel(distances, length_scale, nu=1.5):
    """
    Kernel Matérn para correlación espacial
    nu=1.5 es un buen balance entre suavidad y flexibilidad
    """
    if nu == 0.5:
        # Kernel exponencial
        K = np.exp(-distances / length_scale)
    elif nu == 1.5:
        # Matérn 3/2
        sqrt3_d = np.sqrt(3) * distances / length_scale
        K = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    elif nu == 2.5:
        # Matérn 5/2
        sqrt5_d = np.sqrt(5) * distances / length_scale
        K = (1 + sqrt5_d + (5/3) * (distances / length_scale)**2) * np.exp(-sqrt5_d)
    else:
        raise ValueError(f"nu={nu} no implementado")
    
    return K

def build_spatial_covariance_matrix(coords, length_scale, sigma_sq, nugget=1e-4):
    """
    Construye matriz de covarianza espacial: Σ = σ² * K(d) + nugget * I
    """
    n = coords.shape[0]
    
    # Calcular distancias
    distances = cdist(coords, coords, metric='euclidean')
    
    # Aplicar kernel Matérn
    K = matern_kernel(distances, length_scale, nu=1.5)
    
    # Matriz de covarianza
    Sigma = sigma_sq * K + nugget * np.eye(n)
    
    return Sigma

# ============================================
# 2. RED NEURONAL PROFUNDA
# ============================================
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

# ============================================
# 3. MODELO NN-GLS COMPLETO
# ============================================
class NNGLS_Model:
    """
    Neural Network - Generalized Least Squares para datos espaciales
    Combina:
    - NN para capturar relaciones no lineales
    - GP para modelar correlación espacial de residuos
    """
    
    def __init__(self, input_dim, coords, device='cpu', n_classes=5):
        self.device = device
        self.coords = coords
        self.n_classes = n_classes
        
        # Red neuronal para predecir scores ordinales
        self.nn = DeepNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            output_dim=1  # Score continuo
        ).to(device)
        
        # Parámetros del GP (se optimizan)
        self.log_length_scale = nn.Parameter(torch.tensor(10.0, device=device))
        self.log_sigma_sq = nn.Parameter(torch.tensor(1.0, device=device))
        self.log_nugget = nn.Parameter(torch.tensor(-4.0, device=device))
        
        # Thresholds para clasificación ordinal (n_classes - 1)
        self.thresholds = nn.Parameter(
            torch.linspace(-2, 2, n_classes - 1, device=device)
        )
        
    def compute_spatial_precision(self, coords_batch):
        """
        Calcula matriz de precisión espacial (inversa de covarianza)
        """
        length_scale = torch.exp(self.log_length_scale).item()
        sigma_sq = torch.exp(self.log_sigma_sq).item()
        nugget = torch.exp(self.log_nugget).item()
        
        # Construir matriz de covarianza
        Sigma = build_spatial_covariance_matrix(
            coords_batch, 
            length_scale, 
            sigma_sq, 
            nugget
        )
        
        # Factorización de Cholesky para eficiencia
        try:
            L, lower = cho_factor(Sigma)
            return L, lower, Sigma
        except:
            # Si falla, agregar más nugget
            Sigma += 1e-3 * np.eye(len(Sigma))
            L, lower = cho_factor(Sigma)
            return L, lower, Sigma
    
    def gls_loss(self, predictions, targets, L, lower):
        """
        Loss basado en GLS: (y - μ)^T Σ^(-1) (y - μ)
        """
        residuals = (targets - predictions).cpu().numpy()
        
        # Resolver sistema usando Cholesky: Σ^(-1) residuals
        precision_residuals = cho_solve((L, lower), residuals)
        
        # Calcular loss GLS
        gls_loss = 0.5 * np.dot(residuals, precision_residuals)
        
        return torch.tensor(gls_loss, dtype=torch.float32, device=self.device)
    
    def ordinal_classification_loss(self, scores, labels):
        """
        Loss para clasificación ordinal usando thresholds
        """
        # Ordenar thresholds
        thresholds_sorted = torch.sort(self.thresholds)[0]
        
        # Calcular probabilidades acumulativas
        cum_probs = torch.sigmoid(thresholds_sorted.unsqueeze(0) - scores)
        
        # Agregar extremos (0 y 1)
        cum_probs = torch.cat([
            torch.zeros_like(scores),
            cum_probs,
            torch.ones_like(scores)
        ], dim=1)
        
        # Probabilidades de cada clase
        probs = cum_probs[:, 1:] - cum_probs[:, :-1]
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Negative log likelihood
        nll = F.nll_loss(torch.log(probs), labels)
        
        return nll
    
    def predict_proba(self, X):
        """
        Predice probabilidades de cada clase
        """
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
            
            # Normalizar
            probs = probs / probs.sum(dim=1, keepdim=True)
        
        return probs

# ============================================
# 4. ENTRENAMIENTO NN-GLS
# ============================================
def train_nngls(model, train_loader, coords_train, optimizer, device, 
                use_spatial=True, lambda_spatial=0.5):
    """
    Entrena el modelo NN-GLS combinando loss ordinal y loss espacial
    """
    model.nn.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Predicciones de la NN
        scores = model.nn(features)
        
        # Loss 1: Clasificación ordinal
        ordinal_loss = model.ordinal_classification_loss(scores, labels)
        
        # Loss 2: GLS espacial (solo en mini-batches pequeños por eficiencia)
        spatial_loss = torch.tensor(0.0, device=device)
        if use_spatial and len(features) <= 512:
            # Obtener índices del batch en el dataset completo
            start_idx = batch_idx * len(features)
            end_idx = start_idx + len(features)
            
            if end_idx <= len(coords_train):
                coords_batch = coords_train[start_idx:end_idx]
                
                # Convertir labels a valores ordinales para regresión
                ordinal_values = labels.float().unsqueeze(1)
                
                try:
                    L, lower, _ = model.compute_spatial_precision(coords_batch)
                    spatial_loss = model.gls_loss(scores, ordinal_values, L, lower)
                except:
                    spatial_loss = torch.tensor(0.0, device=device)
        
        # Combinar losses
        total = ordinal_loss + lambda_spatial * spatial_loss
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.nn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([model.thresholds, model.log_length_scale, 
                                       model.log_sigma_sq, model.log_nugget], max_norm=1.0)
        optimizer.step()
        
        total_loss += total.item()
        
        # Predicciones
        with torch.no_grad():
            probs = model.predict_proba(features)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), acc

def validate_nngls(model, val_loader, device):
    """
    Valida el modelo NN-GLS
    """
    model.nn.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Predicciones
            probs = model.predict_proba(features)
            preds = torch.argmax(probs, dim=1)
            
            # Loss ordinal
            scores = model.nn(features)
            loss = model.ordinal_classification_loss(scores, labels)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(val_loader), acc, f1

# ============================================
# 5. PREPARACIÓN DE DATOS
# ============================================
print("=" * 60)
print("NN-GLS: NEURAL NETWORK - GENERALIZED LEAST SQUARES")
print("=" * 60)

print("\n[1/5] Cargando datos...")
df = pd.read_parquet(ruta_archivo)
train_df = df[df['IBD_EQR_Status'].notna()].copy()
test_df = df[df['IBD_EQR_Status'].isna()].copy()

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Selección de features
target_cols = ['IBD', 'IBD_EQR', 'IBD_EQR_Status']
spatial_cols = ['Longitude_Lambert93', 'Latitude_Lambert93', 'Altitude']

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in target_cols]

# Filtrar por NaNs y varianza
threshold_nan = 0.70
numeric_cols = [col for col in numeric_cols 
                if train_df[col].isna().sum() / len(train_df) < threshold_nan]

print(f"Features numericas: {len(numeric_cols)}")

# Preparar X, y, coords
X_train = train_df[numeric_cols].copy()
y_train = train_df['IBD_EQR_Status'].copy()
X_test = test_df[numeric_cols].copy()

coords_train = train_df[spatial_cols].values
coords_test = test_df[spatial_cols].values

# Imputar
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Escalar coordenadas (importante para el GP)
coord_scaler = StandardScaler()
coords_train_scaled = coord_scaler.fit_transform(coords_train)
coords_test_scaled = coord_scaler.transform(coords_test)

# Label encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
n_classes = len(le.classes_)

print(f"Clases: {le.classes_}")
print(f"Shape: X={X_train_scaled.shape}, coords={coords_train_scaled.shape}")

# ============================================
# 6. ENTRENAMIENTO CON K-FOLD CV
# ============================================
print("\n[2/5] Entrenamiento con Validacion Cruzada...")

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train_encoded)):
    print(f"\n  Fold {fold + 1}/{N_FOLDS}")
    
    # Split data
    X_tr = torch.FloatTensor(X_train_scaled[train_idx])
    X_val = torch.FloatTensor(X_train_scaled[val_idx])
    y_tr = torch.LongTensor(y_train_encoded[train_idx])
    y_val = torch.LongTensor(y_train_encoded[val_idx])
    
    coords_tr = coords_train_scaled[train_idx]
    coords_val = coords_train_scaled[val_idx]
    
    # Datasets
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo NN-GLS
    model = NNGLS_Model(
        input_dim=X_train_scaled.shape[1],
        coords=coords_tr,
        device=DEVICE,
        n_classes=n_classes
    )
    
    # Optimizador para NN + parámetros GP
    optimizer = torch.optim.AdamW([
        {'params': model.nn.parameters(), 'lr': LEARNING_RATE, 'weight_decay': 1e-4},
        {'params': [model.thresholds], 'lr': LEARNING_RATE * 0.5},
        {'params': [model.log_length_scale, model.log_sigma_sq, model.log_nugget], 
         'lr': LEARNING_RATE * 0.1}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    # Early stopping
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Usar GLS espacial solo en epochs tardías para estabilidad
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
            print(f"    Epoch {epoch+1:3d}: Train Acc={train_acc:.4f} | "
                  f"Val Acc={val_acc:.4f}, F1={val_f1:.4f} | "
                  f"Length_scale={length_scale:.2f}")
        
        if patience_counter >= patience:
            print(f"    Early stopping en epoch {epoch+1}")
            break
    
    # Restaurar mejor modelo
    model.nn.load_state_dict(best_model_state['nn'])
    model.thresholds = best_model_state['thresholds']
    model.log_length_scale = best_model_state['log_length_scale']
    model.log_sigma_sq = best_model_state['log_sigma_sq']
    model.log_nugget = best_model_state['log_nugget']
    
    fold_scores.append(best_val_acc)
    fold_models.append(model)
    
    print(f"  Mejor Accuracy: {best_val_acc:.4f}")
    print(f"  Parametros GP: length_scale={torch.exp(model.log_length_scale).item():.2f}, "
          f"sigma_sq={torch.exp(model.log_sigma_sq).item():.3f}")

print(f"\nCV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# ============================================
# 7. PREDICCIÓN ESPACIAL EN TEST
# ============================================
print("\n[3/5] Prediccion espacial en test...")

X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

# Ensemble de modelos
all_probs = []
for model in fold_models:
    probs = model.predict_proba(X_test_tensor)
    all_probs.append(probs.cpu().numpy())

# Promediar probabilidades
ensemble_probs = np.mean(all_probs, axis=0)
final_preds = np.argmax(ensemble_probs, axis=1)
y_test_labels = le.inverse_transform(final_preds)

print(f"Predicciones realizadas: {len(y_test_labels)}")

# Distribución
unique, counts = np.unique(y_test_labels, return_counts=True)
print("\nDistribucion de predicciones:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)")

# ============================================
# 8. EVALUACIÓN EN TRAIN
# ============================================
print("\n[4/5] Evaluacion en train...")

X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)

train_probs_all = []
for model in fold_models:
    probs = model.predict_proba(X_train_tensor)
    train_probs_all.append(probs.cpu().numpy())

train_probs = np.mean(train_probs_all, axis=0)
train_preds = np.argmax(train_probs, axis=1)

train_acc = accuracy_score(y_train_encoded, train_preds)
train_f1 = f1_score(y_train_encoded, train_preds, average='weighted')

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Train F1-Score: {train_f1:.4f}")

print("\nReporte detallado:")
print(classification_report(y_train_encoded, train_preds, target_names=le.classes_, digits=4))

# ============================================
# 9. GUARDAR RESULTADOS
# ============================================
print("\n[5/5] Guardando resultados...")

import os
os.makedirs(output_path, exist_ok=True)

results_df = pd.DataFrame({
    'SamplingOperations_code': test_df['SamplingOperations_code'].values,
    'IBD_EQR_Status': y_test_labels
})

output_file = os.path.join(output_path, 'predictions_nngls.csv')
results_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"CSV guardado: predictions_nngls.csv")

# Estadísticas
stats_file = os.path.join(output_path, 'model_stats_nngls.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("NN-GLS: NEURAL NETWORK - GENERALIZED LEAST SQUARES\n")
    f.write("=" * 60 + "\n\n")
    f.write("Metodologia:\n")
    f.write("  - Red Neuronal Profunda (512->256->128->64)\n")
    f.write("  - Gaussian Process con kernel Matern (nu=1.5)\n")
    f.write("  - Clasificacion Ordinal con thresholds aprendidos\n")
    f.write("  - Optimizacion conjunta NN + parametros GP\n\n")
    f.write(f"Configuracion:\n")
    f.write(f"  - Device: {DEVICE}\n")
    f.write(f"  - Folds: {N_FOLDS}\n")
    f.write(f"  - Batch Size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs: {EPOCHS}\n")
    f.write(f"  - Learning Rate: {LEARNING_RATE}\n\n")
    f.write(f"Parametros GP finales (promedio):\n")
    avg_length = np.mean([torch.exp(m.log_length_scale).item() for m in fold_models])
    avg_sigma = np.mean([torch.exp(m.log_sigma_sq).item() for m in fold_models])
    f.write(f"  - Length scale: {avg_length:.3f}\n")
    f.write(f"  - Sigma^2: {avg_sigma:.3f}\n\n")
    f.write(f"Rendimiento:\n")
    f.write(f"  - CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})\n")
    f.write(f"  - Train Accuracy: {train_acc:.4f}\n")
    f.write(f"  - Train F1-Score: {train_f1:.4f}\n\n")
    f.write("Predicciones Test:\n")
    for label, count in zip(unique, counts):
        f.write(f"  {label:12s}: {count:5d} ({count/len(y_test_labels)*100:5.2f}%)\n")

print(f"Estadisticas guardadas: model_stats_nngls.txt")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nArchivos generados:")
print(f"  1. predictions_nngls.csv")
print(f"  2. model_stats_nngls.txt")
print(f"\nCV Accuracy: {np.mean(fold_scores):.2%}")
print(f"Train Accuracy: {train_acc:.2%}")
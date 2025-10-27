import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors
import warnings
import os
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class WaterQualityDataset(Dataset):
    def __init__(self, features, spatial_features, labels=None):
        self.features = torch.FloatTensor(features)
        self.spatial_features = torch.FloatTensor(spatial_features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        spatial = self.spatial_features[idx]
        
        if self.labels is not None:
            return features, spatial, self.labels[idx]
        return features, spatial

class MixupDataset(Dataset):
    """Dataset con Mixup para mejorar generalización"""
    def __init__(self, features, spatial_features, labels, alpha=0.2):
        self.features = torch.FloatTensor(features)
        self.spatial_features = torch.FloatTensor(spatial_features)
        self.labels = torch.LongTensor(labels)
        self.alpha = alpha
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.alpha > 0 and np.random.rand() < 0.5:
            # Aplicar mixup
            lam = np.random.beta(self.alpha, self.alpha)
            idx2 = np.random.randint(0, len(self.features))
            
            mixed_features = lam * self.features[idx] + (1 - lam) * self.features[idx2]
            mixed_spatial = lam * self.spatial_features[idx] + (1 - lam) * self.spatial_features[idx2]
            
            return mixed_features, mixed_spatial, self.labels[idx], self.labels[idx2], lam
        else:
            return self.features[idx], self.spatial_features[idx], self.labels[idx], self.labels[idx], 1.0

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = 1e-7
        
    def forward(self, inputs, targets, mixup_target=None, lam=1.0):
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        # Convertir lam a float si es tensor
        if isinstance(lam, torch.Tensor):
            lam = lam.item() if lam.numel() == 1 else lam.mean().item()
        
        if mixup_target is not None and lam < 1.0:
            # Loss con mixup
            ce_loss1 = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
            ce_loss2 = F.cross_entropy(inputs, mixup_target, reduction='none', label_smoothing=self.label_smoothing)
            ce_loss = lam * ce_loss1 + (1 - lam) * ce_loss2
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        ce_loss = torch.clamp(ce_loss, min=self.eps, max=10)
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_clipped = torch.clamp(self.alpha, min=0.3, max=4.0)
            alpha_t = alpha_clipped[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class MultiHeadSpatialAttention(nn.Module):
    """Multi-Head Attention para features espaciales"""
    def __init__(self, spatial_dim, num_heads=4, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, spatial_features):
        batch_size = spatial_features.size(0)
        
        # Encode spatial features
        x = self.spatial_encoder(spatial_features)
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.bmm(attn_weights, V)
        context = context.view(batch_size, self.hidden_dim)
        
        output = self.out_proj(context)
        
        return output, attn_weights.mean(dim=1)

class ResidualBlock(nn.Module):
    """Residual block con normalización"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return F.relu(x + self.dropout(self.layers(x)))

class FeatureInteractionLayer(nn.Module):
    """Capa para capturar interacciones entre features"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Interacciones de segundo orden (aproximación eficiente)
        interaction = self.interaction(x)
        return interaction

class AdvancedWaterQualityClassifier(nn.Module):
    def __init__(self, input_dim, spatial_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.num_classes = num_classes
        
        # Multi-Head Spatial Attention
        self.spatial_attention = MultiHeadSpatialAttention(spatial_dim, num_heads=4, hidden_dim=128)
        
        # Feature Interaction Layer
        self.feature_interaction = FeatureInteractionLayer(input_dim, hidden_dim=256)
        
        # Deep Feature Encoder con residual connections
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(512, dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            ResidualBlock(256, dropout_rate * 0.7)
        )
        
        # CNN branch mejorado
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(20),
            nn.Flatten()
        )
        
        # Proyección de spatial features a 256 dim para cross-attention
        self.spatial_proj = nn.Linear(128, 256)
        
        # Cross-Attention entre features espaciales y normales
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=dropout_rate * 0.5, batch_first=True
        )
        
        # Proyección de spatial features a 256 dim para cross-attention
        self.spatial_proj = nn.Linear(128, 256)
        
        # Cross-Attention entre features espaciales y normales
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=dropout_rate * 0.5, batch_first=True
        )
        
        # Fusion dimension: feature_encoder(256) + feature_interaction(256) + spatial(128) + cnn(128*20)
        fusion_dim = 256 + 256 + 128 + (128 * 20)
        
        # Advanced Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            ResidualBlock(512, dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            ResidualBlock(256, dropout_rate * 0.3)
        )
        
        # Classifier final con múltiples capas
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features, spatial_features):
        # Procesar features espaciales con multi-head attention
        spatial_encoded, spatial_attn = self.spatial_attention(spatial_features)
        
        # Feature interaction
        feature_interact = self.feature_interaction(features)
        
        # Deep feature encoding
        feature_encoded = self.feature_encoder(features)
        
        # CNN branch
        cnn_input = features.unsqueeze(1)
        cnn_features = self.cnn_branch(cnn_input)
        
        # Cross-attention entre features normales y espaciales
        # Proyectar spatial a 256 dim para que coincida con feature_encoded
        spatial_proj = self.spatial_proj(spatial_encoded)
        
        # Reshape para attention
        feature_for_attn = feature_encoded.unsqueeze(1)
        spatial_for_attn = spatial_proj.unsqueeze(1)
        
        cross_attn_output, _ = self.cross_attention(
            feature_for_attn, spatial_for_attn, spatial_for_attn
        )
        cross_attn_output = cross_attn_output.squeeze(1)
        
        # Fusionar todas las características
        combined = torch.cat([
            cross_attn_output,
            feature_interact,
            spatial_encoded,
            cnn_features
        ], dim=1)
        
        # Fusion layer con residual connections
        fused_features = self.fusion_layer(combined)
        
        # Aplicar spatial attention a las features fusionadas
        spatial_attn_expanded = spatial_attn.mean(dim=1, keepdim=True).expand(-1, fused_features.size(1))
        fused_features = fused_features * (1 + spatial_attn_expanded * 0.3)
        
        # Clasificación final
        output = self.classifier(fused_features)
        
        return output

class AdvancedWaterQualityClassifier_System:
    def __init__(self, use_focal_loss=True, n_folds=6, use_mixup=True):
        self.scaler = RobustScaler()
        self.spatial_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = []
        self.use_focal_loss = use_focal_loss
        self.n_folds = n_folds
        self.use_mixup = use_mixup
        self.class_weights = None
        self.categorical_encoders = {}
        self.feature_names = None
        
        # Para features espaciales
        self.grid_centroids = None
        self.class_centroids = None
        self.knn_model = None
        
    def load_data(self, data_path):
        """Cargar datos desde el archivo parquet"""
        data = pd.read_parquet(data_path)
        
        # Si SamplingOperations_code está como índice, resetearlo
        if data.index.name == 'SamplingOperations_code' or 'SamplingOperations_code' not in data.columns:
            data = data.reset_index()
        
        print(data)
        print(f"\nColumnas disponibles: {list(data.columns)}")
        print(f"Primera columna: {data.columns[0]}")
        
        return data
    
    def create_spatial_features(self, coords, labels=None, is_training=True):
        """Crear features espaciales avanzadas"""
        spatial_features = []
        
        # 1. Coordenadas normalizadas
        spatial_features.append(coords)
        
        if is_training:
            # 2. Grid-based clustering
            n_bins = 20
            
            lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
            lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
            
            lon_bins = np.linspace(lon_min, lon_max, n_bins + 1)
            lat_bins = np.linspace(lat_min, lat_max, n_bins + 1)
            
            lon_indices = np.digitize(coords[:, 0], lon_bins) - 1
            lat_indices = np.digitize(coords[:, 1], lat_bins) - 1
            
            # Calcular centroides de cada celda del grid
            self.grid_centroids = {}
            for i in range(n_bins):
                for j in range(n_bins):
                    mask = (lon_indices == i) & (lat_indices == j)
                    if mask.sum() > 0:
                        self.grid_centroids[(i, j)] = coords[mask].mean(axis=0)
            
            # Distancias a los centroides del grid
            n_closest_grids = min(15, len(self.grid_centroids))
            grid_distances = np.zeros((len(coords), n_closest_grids))
            
            for idx, point in enumerate(coords):
                distances_to_centroids = [
                    np.linalg.norm(point - centroid) 
                    for centroid in self.grid_centroids.values()
                ]
                sorted_distances = sorted(distances_to_centroids)[:n_closest_grids]
                while len(sorted_distances) < n_closest_grids:
                    sorted_distances.append(sorted_distances[-1] if sorted_distances else 0)
                grid_distances[idx] = sorted_distances
            
            spatial_features.append(grid_distances)
            
            # 3. Centroides por clase
            if labels is not None:
                self.class_centroids = {}
                for class_label in np.unique(labels):
                    class_mask = labels == class_label
                    if class_mask.sum() > 0:
                        self.class_centroids[class_label] = coords[class_mask].mean(axis=0)
                
                class_distances = np.zeros((len(coords), len(self.class_centroids)))
                for i, (class_label, centroid) in enumerate(self.class_centroids.items()):
                    distances = np.linalg.norm(coords - centroid, axis=1)
                    class_distances[:, i] = distances
                
                spatial_features.append(class_distances)
            
            # 4. K-nearest neighbors
            k = min(15, max(5, len(coords) // 1000))
            try:
                self.knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=1)
                self.knn_model.fit(coords)
                
                distances, indices = self.knn_model.kneighbors(coords)
                
                knn_features = np.column_stack([
                    distances.mean(axis=1),
                    distances.std(axis=1),
                    distances.min(axis=1),
                    distances.max(axis=1),
                    np.percentile(distances, 25, axis=1),
                    np.percentile(distances, 75, axis=1)
                ])
                spatial_features.append(knn_features)
            except Exception as e:
                print(f"WARNING: KNN features fallaron: {e}")
                dummy_knn = np.zeros((len(coords), 6))
                spatial_features.append(dummy_knn)
                self.knn_model = None
            
        else:  # Test set
            if hasattr(self, 'grid_centroids') and self.grid_centroids:
                n_closest_grids = 15
                grid_distances = np.zeros((len(coords), n_closest_grids))
                
                for idx, point in enumerate(coords):
                    distances_to_centroids = [
                        np.linalg.norm(point - centroid) 
                        for centroid in self.grid_centroids.values()
                    ]
                    sorted_distances = sorted(distances_to_centroids)[:n_closest_grids]
                    while len(sorted_distances) < n_closest_grids:
                        sorted_distances.append(sorted_distances[-1] if sorted_distances else 0)
                    grid_distances[idx] = sorted_distances
                
                spatial_features.append(grid_distances)
            
            if hasattr(self, 'class_centroids') and self.class_centroids:
                class_distances = np.zeros((len(coords), len(self.class_centroids)))
                for i, (class_label, centroid) in enumerate(self.class_centroids.items()):
                    distances = np.linalg.norm(coords - centroid, axis=1)
                    class_distances[:, i] = distances
                spatial_features.append(class_distances)
            
            if self.knn_model is not None:
                try:
                    distances, indices = self.knn_model.kneighbors(coords)
                    knn_features = np.column_stack([
                        distances.mean(axis=1),
                        distances.std(axis=1),
                        distances.min(axis=1),
                        distances.max(axis=1),
                        np.percentile(distances, 25, axis=1),
                        np.percentile(distances, 75, axis=1)
                    ])
                    spatial_features.append(knn_features)
                except:
                    dummy_knn = np.zeros((len(coords), 6))
                    spatial_features.append(dummy_knn)
        
        all_spatial_features = np.hstack(spatial_features)
        return all_spatial_features
    
    def preprocess_categorical_variables(self, df):
        """Convertir variables categóricas usando encoding"""
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns 
                             if col not in ['SamplingOperations_code', 'IBD_EQR_Status']]
        
        print(f"Procesando {len(categorical_columns)} variables categóricas...")
        
        for col in categorical_columns:
            df_processed[col] = df_processed[col].fillna('_MISSING_').astype(str)
            
            if col not in self.categorical_encoders:
                le = LabelEncoder()
                le.fit(df_processed[col])
                df_processed[col] = le.transform(df_processed[col])
                self.categorical_encoders[col] = le
            else:
                known_classes = set(self.categorical_encoders[col].classes_)
                fallback_value = list(known_classes)[0]
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_classes else fallback_value
                )
                df_processed[col] = self.categorical_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def clean_and_prepare_features(self, df, is_training=True):
        """Limpiar y preparar características"""
        print("Limpieza y preparación de características...")
        
        spatial_cols = ['Longitude_Lambert93', 'Latitude_Lambert93']
        
        exclude_cols = ['SamplingOperations_code', 'IBD_EQR_Status'] + spatial_cols
        if 'IBD' in df.columns:
            exclude_cols.append('IBD')
        if 'IBD_EQR' in df.columns:
            exclude_cols.append('IBD_EQR')
        
        coords = df[spatial_cols].values
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        df_processed = self.preprocess_categorical_variables(df)
        X = df_processed[feature_columns].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if is_training:
            nan_threshold = 0.65
            self.high_nan_cols = X.columns[X.isnull().mean() > nan_threshold].tolist()
            if self.high_nan_cols:
                print(f"  Eliminando {len(self.high_nan_cols)} columnas con >{nan_threshold*100}% NaN")
                X = X.drop(columns=self.high_nan_cols)
            
            self.feature_medians = {}
            for col in X.select_dtypes(include=[np.number]).columns:
                self.feature_medians[col] = X[col].median()
                if X[col].isnull().any():
                    X[col] = X[col].fillna(self.feature_medians[col])
            
            self.variance_threshold = VarianceThreshold(threshold=1e-6)
            X_var_filtered = self.variance_threshold.fit_transform(X)
            selected_features = X.columns[self.variance_threshold.get_support()]
            X = X[selected_features]
            self.feature_names = list(selected_features)
            print(f"  Features después del filtro: {X.shape[1]}")
        else:
            if hasattr(self, 'high_nan_cols') and self.high_nan_cols:
                X = X.drop(columns=[col for col in self.high_nan_cols if col in X.columns])
            
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in self.feature_medians:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(self.feature_medians[col])
                else:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(0)
            
            if hasattr(self, 'feature_names') and self.feature_names:
                missing_features = [col for col in self.feature_names if col not in X.columns]
                if missing_features:
                    for col in missing_features:
                        X[col] = 0
                X = X[self.feature_names]
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        final_X = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_X, coords
    
    def preprocess_data(self, data):
        """Preprocesar los datos completos"""
        print("\n" + "="*80)
        print("PREPROCESAMIENTO DE DATOS")
        print("="*80)
        
        train_mask = data['IBD_EQR_Status'].notna()
        train_data = data[train_mask].copy()
        test_data = data[~train_mask].copy()
        
        print(f"Datos de entrenamiento: {len(train_data):,}")
        print(f"Datos de prueba: {len(test_data):,}")
        
        X_train, coords_train = self.clean_and_prepare_features(train_data, is_training=True)
        y_train = train_data['IBD_EQR_Status'].values
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        X_test, coords_test = self.clean_and_prepare_features(test_data, is_training=False)
        
        if X_test.shape[1] != X_train.shape[1]:
            min_features = min(X_train.shape[1], X_test.shape[1])
            X_train = X_train[:, :min_features]
            X_test = X_test[:, :min_features]
        
        print(f"\nEscalando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Creando features espaciales...")
        spatial_train = self.create_spatial_features(coords_train, y_train_encoded, is_training=True)
        spatial_test = self.create_spatial_features(coords_test, is_training=False)
        
        spatial_train_scaled = self.spatial_scaler.fit_transform(spatial_train)
        spatial_test_scaled = self.spatial_scaler.transform(spatial_test)
        
        print(f"\nDimensiones finales:")
        print(f"  X_train: {X_train_scaled.shape}")
        print(f"  spatial_train: {spatial_train_scaled.shape}")
        print(f"  X_test: {X_test_scaled.shape}")
        print(f"  spatial_test: {spatial_test_scaled.shape}")
        
        if self.use_focal_loss:
            unique_classes = np.unique(y_train_encoded)
            raw_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_encoded)
            raw_weights = np.clip(raw_weights, 0.5, 4.0)
            raw_weights = np.sqrt(raw_weights)
            self.class_weights = torch.FloatTensor(raw_weights).to(device)
            
            print(f"\nClass weights:")
            for cls, weight in zip(self.label_encoder.classes_, raw_weights):
                print(f"  {cls}: {weight:.3f}")
        
        return (X_train_scaled, spatial_train_scaled, X_test_scaled, spatial_test_scaled, 
                y_train_encoded, test_data[['SamplingOperations_code']])
    
    def create_weighted_sampler(self, y_train):
        """Crear sampler balanceado"""
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_fold(self, X_train_fold, spatial_train_fold, y_train_fold, 
                   X_val_fold, spatial_val_fold, y_val_fold, fold_idx, epochs=200):
        """Entrenar un fold individual"""
        
        if self.use_mixup:
            train_dataset = MixupDataset(X_train_fold, spatial_train_fold, y_train_fold, alpha=0.2)
        else:
            train_dataset = WaterQualityDataset(X_train_fold, spatial_train_fold, y_train_fold)
        
        val_dataset = WaterQualityDataset(X_val_fold, spatial_val_fold, y_val_fold)
        
        weighted_sampler = self.create_weighted_sampler(y_train_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=weighted_sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        input_dim = X_train_fold.shape[1]
        spatial_dim = spatial_train_fold.shape[1]
        num_classes = len(np.unique(y_train_fold))
        
        model = AdvancedWaterQualityClassifier(
            input_dim, spatial_dim, num_classes, dropout_rate=0.3
        ).to(device)
        
        if self.use_focal_loss:
            criterion = ImprovedFocalLoss(alpha=self.class_weights, gamma=2.5, label_smoothing=0.05)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.05)
        
        # Optimizer con learning rate warmup
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5, betas=(0.9, 0.999))
        
        # Cosine annealing con warmup
        warmup_epochs = 10
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        best_f1 = 0.0
        best_val_acc = 0.0
        patience = 30
        patience_counter = 0
        best_model_state = None
        
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train F1':>9} | {'Val F1':>9} | {'Val Acc':>9} | {'LR':>10}")
        print("-" * 70)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            all_predictions = []
            all_targets = []
            
            for batch_data in train_loader:
                if self.use_mixup and len(batch_data) == 5:
                    batch_features, batch_spatial, batch_labels, mixup_labels, lam = batch_data
                else:
                    batch_features, batch_spatial, batch_labels = batch_data[:3]
                    mixup_labels = None
                    lam = 1.0
                
                batch_features = batch_features.to(device)
                batch_spatial = batch_spatial.to(device)
                batch_labels = batch_labels.to(device)
                
                if mixup_labels is not None:
                    mixup_labels = mixup_labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_features, batch_spatial)
                
                if self.use_focal_loss and self.use_mixup:
                    loss = criterion(outputs, batch_labels, mixup_labels, lam)
                else:
                    loss = criterion(outputs, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
            
            # Validación
            val_f1, val_acc = self.evaluate_fold(model, val_loader)
            
            if len(all_predictions) > 0:
                train_f1 = f1_score(all_targets, all_predictions, average='weighted')
            else:
                train_f1 = 0.0
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar mejor modelo
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 5:
                avg_loss = total_loss / len(train_loader)
                print(f"{epoch:5d} | {avg_loss:10.4f} | {train_f1:9.4f} | {val_f1:9.4f} | {val_acc:9.4f} | {current_lr:10.2e}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping en época {epoch}")
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"Fold {fold_idx+1} completado - Mejor F1: {best_f1:.4f}, Mejor Acc: {best_val_acc:.4f}")
        
        return model, best_f1, best_val_acc
    
    def evaluate_fold(self, model, data_loader):
        """Evaluar un fold"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_spatial, batch_labels in data_loader:
                batch_features = batch_features.to(device)
                batch_spatial = batch_spatial.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features, batch_spatial)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
        
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        acc = accuracy_score(all_targets, all_predictions)
        
        return f1, acc
    
    def train_model(self, X_train, spatial_train, y_train, epochs=200):
        """Entrenar modelo con validación cruzada estratificada"""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_f1_scores = []
        fold_acc_scores = []
        
        print("\n" + "="*80)
        print(f"ENTRENAMIENTO CON {self.n_folds}-FOLD CROSS VALIDATION")
        print("="*80)
        print(f"Distribución de clases: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*80}")
            
            X_train_fold = X_train[train_idx]
            spatial_train_fold = spatial_train[train_idx]
            y_train_fold = y_train[train_idx]
            
            X_val_fold = X_train[val_idx]
            spatial_val_fold = spatial_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            model, fold_f1, fold_acc = self.train_fold(
                X_train_fold, spatial_train_fold, y_train_fold,
                X_val_fold, spatial_val_fold, y_val_fold,
                fold_idx, epochs
            )
            
            self.models.append(model)
            fold_f1_scores.append(fold_f1)
            fold_acc_scores.append(fold_acc)
        
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        mean_acc = np.mean(fold_acc_scores)
        std_acc = np.std(fold_acc_scores)
        
        print("\n" + "="*80)
        print("RESULTADOS CROSS-VALIDATION")
        print("="*80)
        print(f"F1 Score:  {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"\nF1 por fold:  {[f'{f1:.4f}' for f1 in fold_f1_scores]}")
        print(f"Acc por fold: {[f'{acc:.4f}' for acc in fold_acc_scores]}")
        
        return mean_f1, mean_acc
    
    def predict(self, X_test, spatial_test, test_info):
        """Generar predicciones usando ensemble"""
        print("\n" + "="*80)
        print("GENERANDO PREDICCIONES CON ENSEMBLE")
        print("="*80)
        
        all_fold_predictions = []
        
        for fold_idx, model in enumerate(self.models):
            model.eval()
            
            test_dataset = WaterQualityDataset(X_test, spatial_test)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
            
            fold_predictions = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_features, batch_spatial = batch_data[0], batch_data[1]
                    
                    batch_features = batch_features.to(device)
                    batch_spatial = batch_spatial.to(device)
                    
                    outputs = model(batch_features, batch_spatial)
                    
                    # Temperature scaling para mejor calibración
                    temperature = 1.2
                    outputs_scaled = outputs / temperature
                    probabilities = F.softmax(outputs_scaled, dim=1)
                    
                    fold_predictions.append(probabilities.cpu().numpy())
            
            all_fold_predictions.append(np.vstack(fold_predictions))
            print(f"  Fold {fold_idx+1}/{len(self.models)} procesado")
        
        # Ensemble con promedio ponderado
        weights = np.array([1.0 + i * 0.15 for i in range(len(all_fold_predictions))])
        weights = weights / weights.sum()
        
        final_predictions = np.average(all_fold_predictions, axis=0, weights=weights)
        
        predicted_classes = np.argmax(final_predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        # Crear DataFrame con TODAS las columnas importantes
        results_df = test_info.copy()
        results_df['IBD_EQR_Status'] = predicted_labels
        
        # Añadir probabilidades para cada clase
        class_names = self.label_encoder.classes_
        for i, cls in enumerate(class_names):
            results_df[f'Probability_{cls}'] = final_predictions[:, i]
        
        # Añadir confianza de la predicción
        results_df['Prediction_Confidence'] = final_predictions.max(axis=1)
        
        # Añadir segunda opción más probable
        second_best_idx = np.argsort(final_predictions, axis=1)[:, -2]
        second_best_labels = self.label_encoder.inverse_transform(second_best_idx)
        results_df['Second_Best_Prediction'] = second_best_labels
        results_df['Second_Best_Probability'] = final_predictions[np.arange(len(final_predictions)), second_best_idx]
        
        return results_df

def main():
    print("\n" + "="*80)
    print("CLASIFICADOR AVANZADO DE CALIDAD DEL AGUA")
    print("Con Arquitectura Deep Learning de Alta Precisión")
    print("="*80)
    
    classifier = AdvancedWaterQualityClassifier_System(
        use_focal_loss=True, 
        n_folds=6, 
        use_mixup=True
    )
    
    # CAMBIAR ESTAS RUTAS
    data_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\data\processed\03_CLEAN_COMPLETE_DF_02.parquet"
    output_dir = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\008_advanced_nn"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n[PASO 1/4] Cargando datos...")
        data = classifier.load_data(data_path)
        
        print("\n[PASO 2/4] Preprocesando datos...")
        (X_train, spatial_train, X_test, spatial_test, 
         y_train, test_info) = classifier.preprocess_data(data)
        
        print("\n[PASO 3/4] Entrenando modelo...")
        mean_f1, mean_acc = classifier.train_model(X_train, spatial_train, y_train, epochs=200)
        
        print("\n[PASO 4/4] Generando predicciones...")
        predictions_df = classifier.predict(X_test, spatial_test, test_info)
        
        # Guardar resultados
        output_file = os.path.join(output_dir, "advanced_predictions.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"F1 Score CV: {mean_f1:.4f}")
        print(f"Accuracy CV: {mean_acc:.4f}")
        print(f"Predicciones guardadas en: {output_file}")
        
        print(f"\n📊 DISTRIBUCIÓN DE PREDICCIONES:")
        prediction_counts = predictions_df['IBD_EQR_Status'].value_counts().sort_index()
        for clase, count in prediction_counts.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"  {clase:10s}: {count:5,} muestras ({percentage:5.1f}%)")
        
        print(f"\n🎯 ANÁLISIS DE CONFIANZA:")
        confidence = predictions_df['Prediction_Confidence']
        print(f"  Confianza promedio: {confidence.mean():.3f}")
        print(f"  Confianza mediana:  {confidence.median():.3f}")
        print(f"  Alta confianza (>0.8):   {(confidence > 0.8).sum():,} ({(confidence > 0.8).sum()/len(confidence)*100:.1f}%)")
        print(f"  Media confianza (0.6-0.8): {((confidence >= 0.6) & (confidence <= 0.8)).sum():,}")
        print(f"  Baja confianza (<0.6):   {(confidence < 0.6).sum():,} ({(confidence < 0.6).sum()/len(confidence)*100:.1f}%)")
        
        print(f"\n📁 COLUMNAS EN EL CSV:")
        print(f"  Total de columnas: {len(predictions_df.columns)}")
        print(f"  Columnas: {list(predictions_df.columns)}")
        
        return classifier, predictions_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    classifier, predictions = main()
    
    if classifier is not None and predictions is not None:
        print("\n" + "="*80)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ EL PROCESO FALLÓ")
        print("="*80)
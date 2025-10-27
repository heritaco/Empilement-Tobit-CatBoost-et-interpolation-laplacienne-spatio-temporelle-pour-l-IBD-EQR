import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold
import warnings
import os
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WaterQualityDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        
        if self.labels is not None:
            return features, self.labels[idx]
        return features

class StableFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = 1e-8
        
    def forward(self, inputs, targets):
        # Clip inputs para evitar overflow
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / n_classes
            log_probs = torch.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(targets_one_hot * log_probs, dim=1)
        else:
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # Clamp ce_loss para evitar overflow en exp
        ce_loss = torch.clamp(ce_loss, min=self.eps, max=10)
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=self.eps, max=1.0)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Aplicar alpha weights más conservadores
        if self.alpha is not None:
            if self.label_smoothing > 0:
                alpha_t = torch.mean(self.alpha)
            else:
                # Clip alpha weights para evitar valores extremos
                alpha_clipped = torch.clamp(self.alpha, min=0.1, max=3.0)
                alpha_t = alpha_clipped[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FastFeatureSelector(nn.Module):
    def __init__(self, input_dim, selection_ratio=0.75):
        super().__init__()
        self.selection_ratio = selection_ratio
        # Simplificado: solo 2 capas en lugar de 3
        self.gate = nn.Sequential(
            nn.Linear(input_dim, max(64, input_dim // 8)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, input_dim // 8), input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.gate(x)
        k = int(x.size(1) * self.selection_ratio)
        _, top_indices = torch.topk(attention_weights, k, dim=1)
        
        mask = torch.zeros_like(attention_weights)
        mask.scatter_(1, top_indices, 1)
        
        selected_features = x * attention_weights * mask
        return selected_features, attention_weights

class SimplifiedCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # CNN más simple con menos capas
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.conv_layers(x)

class FastLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # LSTM más simple
        self.seq_len = min(20, max(10, input_dim // 30))
        self.feature_dim = input_dim // self.seq_len
        
        # Solo una capa LSTM bidireccional
        self.lstm = nn.LSTM(self.feature_dim, 64, batch_first=True, bidirectional=True, dropout=0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x[:, :self.seq_len * self.feature_dim]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        x, _ = self.lstm(x)
        
        # Pooling simple
        max_pool = torch.max(x, dim=1)[0]
        avg_pool = torch.mean(x, dim=1)
        
        return torch.cat([max_pool, avg_pool], dim=1)

class FastWaterQualityClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature selector simplificado
        self.feature_selector = FastFeatureSelector(input_dim, selection_ratio=0.75)
        
        # Rama principal más simple
        self.main_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Solo CNN y LSTM, sin Transformer
        self.cnn_branch = SimplifiedCNN(input_dim)
        self.lstm_branch = FastLSTM(input_dim)
        
        # Dimensiones reducidas
        fusion_dim = 128 + (64 * 16) + 256  # main + cnn + lstm
        
        # Clasificador más simple
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
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
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        selected_features, feature_attention = self.feature_selector(x)
        
        main_features = self.main_branch(selected_features)
        cnn_features = self.cnn_branch(selected_features)
        lstm_features = self.lstm_branch(selected_features)
        
        # Concatenar todas las características
        all_features = torch.cat([main_features, cnn_features, lstm_features], dim=1)
        
        output = self.classifier(all_features)
        
        if return_attention:
            return output, feature_attention
        return output

class WaterQualityFastClassifier:
    def __init__(self, use_focal_loss=True, n_folds=4):  # Reducido de 6 a 4 folds
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.models = []
        self.use_focal_loss = use_focal_loss
        self.n_folds = n_folds
        self.class_weights = None
        self.categorical_encoders = {}
        self.feature_names = None
        
    def load_data(self, data_path):
        """Cargar datos desde el archivo parquet"""
        data = pd.read_parquet(data_path)
        return data
    
    def preprocess_categorical_variables(self, df):
        """Convertir variables categóricas usando encoding robusto"""
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns 
                            if col not in ['SamplingOperations_code', 'IBD_EQR_Status']]
        
        print(f"Procesando {len(categorical_columns)} variables categoricas...")
        
        for col in categorical_columns:
            if col not in self.categorical_encoders:
                # Entrenamiento
                le = LabelEncoder()
                # Convertir NaN a string y ajustar
                df_processed[col] = df_processed[col].astype(str)
                le.fit(df_processed[col])
                df_processed[col] = le.transform(df_processed[col])
                self.categorical_encoders[col] = le
                # Guardar el valor por defecto (el más frecuente)
                self.default_categorical_values = getattr(self, 'default_categorical_values', {})
                self.default_categorical_values[col] = 0  # índice de la primera clase
            else:
                # Test: manejar valores no vistos
                df_processed[col] = df_processed[col].astype(str)
                le = self.categorical_encoders[col]
                
                # Crear mapeo seguro
                encoded_values = []
                for val in df_processed[col]:
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        # Usar valor por defecto para clases no vistas
                        encoded_values.append(self.default_categorical_values.get(col, 0))
                
                df_processed[col] = encoded_values
        
        return df_processed
    
    def clean_and_prepare_features(self, df, is_training=True):
        """Limpiar y preparar características"""
        print("Iniciando limpieza y preparacion de caracteristicas...")
        
        exclude_cols = ['SamplingOperations_code', 'IBD_EQR_Status']
        if 'IBD_EQR_Status' in df.columns:
            exclude_cols.append('IBD_EQR_Status')
        
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        df_processed = self.preprocess_categorical_variables(df)
        X = df_processed[feature_columns].copy()
        
        print("Manejando valores infinitos y NaN...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if is_training:
            # Guardar información de columnas problemáticas para usar en test
            nan_threshold = 0.7
            self.high_nan_cols = X.columns[X.isnull().mean() > nan_threshold].tolist()
            if self.high_nan_cols:
                print(f"Eliminando {len(self.high_nan_cols)} columnas con >70% NaN")
                X = X.drop(columns=self.high_nan_cols)
            
            # Guardar medianas para imputación en test
            self.feature_medians = {}
            for col in X.select_dtypes(include=[np.number]).columns:
                self.feature_medians[col] = X[col].median()
                if X[col].isnull().any():
                    X[col] = X[col].fillna(self.feature_medians[col])
        else:
            # Aplicar las mismas transformaciones que en entrenamiento
            if hasattr(self, 'high_nan_cols') and self.high_nan_cols:
                X = X.drop(columns=[col for col in self.high_nan_cols if col in X.columns])
            
            # Imputar usando las medianas calculadas en entrenamiento
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in self.feature_medians:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(self.feature_medians[col])
                else:
                    # Si la columna no existía en train, rellenar con 0
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(0)
        
        if is_training:
            print("Aplicando filtro de varianza...")
            self.variance_threshold = VarianceThreshold(threshold=1e-6)
            X_var_filtered = self.variance_threshold.fit_transform(X)
            selected_features = X.columns[self.variance_threshold.get_support()]
            X = X[selected_features]
            self.feature_names = list(selected_features)
            print(f"Caracteristicas despues del filtro de varianza: {X.shape[1]}")
        else:
            if hasattr(self, 'feature_names') and self.feature_names:
                # Asegurar que tenemos exactamente las mismas features
                available_features = [col for col in self.feature_names if col in X.columns]
                missing_features = [col for col in self.feature_names if col not in X.columns]
                
                if missing_features:
                    print(f"WARNING: {len(missing_features)} features faltantes en test, rellenando con 0")
                    for col in missing_features:
                        X[col] = 0
                
                X = X[self.feature_names]
        
        if is_training and X.shape[1] > 50:
            print("Eliminando caracteristicas altamente correlacionadas...")
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            self.high_corr_features = [column for column in upper_tri.columns 
                                     if any(upper_tri[column] > 0.95)]
            
            if self.high_corr_features:
                print(f"Eliminando {len(self.high_corr_features)} caracteristicas altamente correlacionadas")
                X = X.drop(columns=self.high_corr_features)
                self.feature_names = list(X.columns)
        else:
            # Aplicar la misma eliminación de correlación en test
            if hasattr(self, 'high_corr_features') and self.high_corr_features:
                X = X.drop(columns=[col for col in self.high_corr_features if col in X.columns])
                self.feature_names = [col for col in self.feature_names if col not in self.high_corr_features]
        
        # Verificación final de NaN/Inf
        nan_count = np.isnan(X.values).sum()
        inf_count = np.isinf(X.values).sum()
        
        if nan_count > 0:
            print(f"WARNING: {nan_count} valores NaN encontrados después del preprocesamiento")
            X = X.fillna(0)
        
        if inf_count > 0:
            print(f"WARNING: {inf_count} valores Inf encontrados después del preprocesamiento")
            X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Caracteristicas finales: {X.shape[1]}")
        
        # Verificación final
        final_X = X.values
        if np.isnan(final_X).any() or np.isinf(final_X).any():
            print("ERROR: Todavía hay NaN/Inf después de la limpieza, forzando reemplazo")
            final_X = np.nan_to_num(final_X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_X
    
    def preprocess_data(self, data):
        """Preprocesar los datos completos"""
        print("Iniciando preprocesamiento de datos...")
        
        train_mask = data['IBD_EQR_Status'].notna()
        
        train_data = data[train_mask].copy()
        test_data = data[~train_mask].copy()
        
        print(f"Datos de entrenamiento: {len(train_data)}")
        print(f"Datos de prueba: {len(test_data)}")
        
        X_train = self.clean_and_prepare_features(train_data, is_training=True)
        y_train = train_data['IBD_EQR_Status'].values
        
        X_test = self.clean_and_prepare_features(test_data, is_training=False)
        
        if X_test.shape[1] != X_train.shape[1]:
            min_features = min(X_train.shape[1], X_test.shape[1])
            X_train = X_train[:, :min_features]
            X_test = X_test[:, :min_features]
            print(f"Ajustadas dimensiones a: {min_features}")
        
        print("Aplicando escalado robusto...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if self.use_focal_loss:
            unique_classes = np.unique(y_train_encoded)
            raw_weights = compute_class_weight(
                'balanced', classes=unique_classes, y=y_train_encoded
            )
            # Limitar los pesos extremos para evitar inestabilidad
            raw_weights = np.clip(raw_weights, 0.5, 5.0)
            # Suavizar los pesos
            raw_weights = np.sqrt(raw_weights)
            self.class_weights = torch.FloatTensor(raw_weights).to(device)
            
            print(f"Class weights aplicados: {dict(zip(self.label_encoder.classes_, raw_weights))}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, test_data['SamplingOperations_code']
    
    def create_weighted_sampler(self, y_train):
        """Crear sampler balanceado para entrenamiento"""
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_fold(self, X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx, epochs=120):
        """Entrenar un fold individual"""
        train_dataset = WaterQualityDataset(X_train_fold, y_train_fold)
        val_dataset = WaterQualityDataset(X_val_fold, y_val_fold)
        
        weighted_sampler = self.create_weighted_sampler(y_train_fold)
        
        # Batch size más grande para estabilidad
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=weighted_sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        input_dim = X_train_fold.shape[1]
        num_classes = len(np.unique(y_train_fold))
        
        model = FastWaterQualityClassifier(input_dim, num_classes, dropout_rate=0.3).to(device)
        
        if self.use_focal_loss:
            criterion = StableFocalLoss(alpha=self.class_weights, gamma=1.5, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        
        # Learning rate mucho más conservador
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, 
            min_lr=1e-6, verbose=False
        )
        
        best_f1 = 0.0
        patience = 20
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            all_predictions = []
            all_targets = []
            
            # Verificar estabilidad durante entrenamiento
            nan_detected = False
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Verificar NaN en inputs
                if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                    print(f"WARNING: NaN/Inf detectado en features del batch")
                    continue
                
                optimizer.zero_grad()
                
                outputs = model(batch_features)
                
                # Verificar NaN en outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"WARNING: NaN/Inf detectado en outputs, saltando batch")
                    nan_detected = True
                    continue
                
                loss = criterion(outputs, batch_labels)
                
                # Verificar NaN en loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf detectado en loss, saltando batch")
                    nan_detected = True
                    continue
                
                loss.backward()
                
                # Gradient clipping más agresivo
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Verificar gradientes
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        if torch.isnan(p.grad).any():
                            print("WARNING: NaN en gradientes detectado")
                            nan_detected = True
                            break
                
                if not nan_detected:
                    optimizer.step()
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_labels.cpu().numpy())
            
            if nan_detected:
                print(f"Fold {fold_idx+1} - Epoca {epoch}: NaN detectado, reiniciando pesos...")
                # Reinicializar pesos del modelo
                model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                continue
            
            val_f1 = self.evaluate_fold(model, val_loader)
            
            if len(all_predictions) > 0:
                train_f1 = f1_score(all_targets, all_predictions, average='weighted')
            else:
                train_f1 = 0.0
            
            scheduler.step(val_f1)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 3:
                avg_loss = total_loss / max(len(train_loader), 1)
                print(f"Fold {fold_idx+1} - Epoca {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Train_F1={train_f1:.4f}, Val_F1={val_f1:.4f}")
            
            if patience_counter >= patience:
                print(f"Fold {fold_idx+1} - Early stopping en epoca {epoch}")
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_f1
    
    def evaluate_fold(self, model, data_loader):
        """Evaluar un fold"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
        
        return f1_score(all_targets, all_predictions, average='weighted')
    
    def train_model(self, X_train, y_train, epochs=120):
        """Entrenar modelo con validación cruzada estratificada"""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_f1_scores = []
        
        print(f"Iniciando entrenamiento con {self.n_folds}-Fold Cross Validation...")
        print(f"Distribucion de clases: {np.bincount(y_train)}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n=== FOLD {fold_idx + 1}/{self.n_folds} ===")
            
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            model, fold_f1 = self.train_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                fold_idx, epochs
            )
            
            self.models.append(model)
            fold_f1_scores.append(fold_f1)
            
            print(f"Fold {fold_idx+1} completado - F1 Score: {fold_f1:.4f}")
        
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        
        print(f"\nCross-Validation completado!")
        print(f"F1 Score promedio: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"F1 Scores por fold: {[f'{f1:.4f}' for f1 in fold_f1_scores]}")
        
        return mean_f1
    
    def predict(self, X_test, sample_codes):
        """Generar predicciones usando ensemble de modelos"""
        print("Generando predicciones con ensemble de modelos...")
        
        all_fold_predictions = []
        
        for fold_idx, model in enumerate(self.models):
            model.eval()
            
            test_dataset = WaterQualityDataset(X_test)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
            
            fold_predictions = []
            
            with torch.no_grad():
                for batch_features in test_loader:
                    if isinstance(batch_features, tuple):
                        batch_features = batch_features[0]
                    
                    batch_features = batch_features.to(device)
                    
                    # Verificar NaN en inputs de test
                    if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                        print(f"WARNING: NaN/Inf detectado en test features")
                        # Crear predicciones por defecto (distribución uniforme)
                        batch_size = batch_features.size(0)
                        num_classes = len(self.label_encoder.classes_)
                        default_probs = torch.ones(batch_size, num_classes) / num_classes
                        fold_predictions.append(default_probs.numpy())
                        continue
                    
                    outputs = model(batch_features)
                    
                    # Verificar NaN en outputs de test
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"WARNING: NaN/Inf detectado en test outputs")
                        # Crear predicciones por defecto
                        batch_size = outputs.size(0)
                        num_classes = outputs.size(1)
                        default_probs = torch.ones(batch_size, num_classes) / num_classes
                        fold_predictions.append(default_probs.numpy())
                        continue
                    
                    # Aplicar softmax con clipping para estabilidad
                    outputs_clipped = torch.clamp(outputs, min=-10, max=10)
                    probabilities = torch.softmax(outputs_clipped, dim=1)
                    
                    # Verificar NaN en probabilidades
                    if torch.isnan(probabilities).any():
                        print(f"WARNING: NaN detectado en probabilidades")
                        batch_size = probabilities.size(0)
                        num_classes = probabilities.size(1)
                        probabilities = torch.ones(batch_size, num_classes) / num_classes
                    
                    fold_predictions.append(probabilities.cpu().numpy())
            
            if fold_predictions:
                all_fold_predictions.append(np.vstack(fold_predictions))
            else:
                print(f"WARNING: Fold {fold_idx} no generó predicciones válidas")
        
        if not all_fold_predictions:
            print("ERROR: Ningún fold generó predicciones válidas, usando distribución uniforme")
            num_samples = len(sample_codes)
            num_classes = len(self.label_encoder.classes_)
            final_predictions = np.ones((num_samples, num_classes)) / num_classes
        else:
            # Ensemble: Promedio de predicciones de todos los folds
            final_predictions = np.mean(all_fold_predictions, axis=0)
            
            # Verificar NaN en predicciones finales
            if np.isnan(final_predictions).any():
                print("WARNING: NaN detectado en predicciones finales, usando distribución uniforme")
                num_samples, num_classes = final_predictions.shape
                final_predictions = np.ones((num_samples, num_classes)) / num_classes
        
        predicted_classes = np.argmax(final_predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        results_df = pd.DataFrame({
            'SamplingOperations_code': sample_codes,
            'IBD_EQR_Status': predicted_labels
        })
        
        class_names = self.label_encoder.classes_
        prob_df = pd.DataFrame(final_predictions, columns=[f'prob_{cls}' for cls in class_names])
        results_df = pd.concat([results_df, prob_df], axis=1)
        
        return results_df

def main():
    print("Iniciando Clasificador de Calidad del Agua...")
    classifier = WaterQualityFastClassifier(use_focal_loss=True, n_folds=4)
    
    data_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\data\processed\03_CLEAN_COMPLETE_DF_02.parquet"
    output_dir = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Cargando datos...")
        data = classifier.load_data(data_path)
        
        print("Preprocesando datos...")
        X_train, X_test, y_train, sample_codes = classifier.preprocess_data(data)
        
        print(f"Dimensiones finales:")
        print(f"  - Entrenamiento: {X_train.shape}")
        print(f"  - Prueba: {X_test.shape}")
        print(f"  - Clases unicas: {len(np.unique(y_train))}")
        
        print("Iniciando entrenamiento...")
        mean_f1 = classifier.train_model(X_train, y_train, epochs=120)
        
        print("Generando predicciones...")
        predictions_df = classifier.predict(X_test, sample_codes)
        
        output_file = os.path.join(output_dir, "water_quality_predictions.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print(f"\nENTRENAMIENTO COMPLETADO!")
        print(f"F1 Score promedio CV: {mean_f1:.4f}")
        print(f"Predicciones guardadas en: {output_file}")
        
        print(f"\nDistribucion de predicciones:")
        prediction_counts = predictions_df['IBD_EQR_Status'].value_counts().sort_index()
        for clase, count in prediction_counts.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"  {clase}: {count:,} muestras ({percentage:.1f}%)")
        
        prob_columns = [col for col in predictions_df.columns if col.startswith('prob_')]
        if prob_columns:
            max_probs = predictions_df[prob_columns].max(axis=1)
            avg_confidence = max_probs.mean()
            low_confidence = (max_probs < 0.6).sum()
            
            print(f"\nAnalisis de confianza:")
            print(f"  Confianza promedio: {avg_confidence:.3f}")
            print(f"  Predicciones con baja confianza (<0.6): {low_confidence:,}")
            print(f"  Confianza minima: {max_probs.min():.3f}")
            print(f"  Confianza maxima: {max_probs.max():.3f}")
        
        return classifier, predictions_df
        
    except Exception as e:
        print(f"Error durante la ejecucion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    classifier, predictions = main()
    if classifier is not None and predictions is not None:
        print("\nCLASIFICACION DE CALIDAD DEL AGUA COMPLETADA EXITOSAMENTE!")
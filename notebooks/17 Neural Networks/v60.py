import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostClassifier, Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\notebooks\06_cb_regression\completo\tp_full.parquet"
output_path = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\results\006 NN\predictions_IBD_EQR_Status.csv"

df = pd.read_parquet(ruta_archivo)

target_col = 'IBD_EQR_Status'
coord_cols = ['Longitude_Lambert93', 'Latitude_Lambert93']
id_col = 'SamplingOperations_code'

df_train = df[df[target_col].notna()].copy()
df_pred = df[df[target_col].isna()].copy()

for col in df_train.columns:
    if col not in [target_col, id_col] + coord_cols:
        if df_train[col].dtype == 'object':
            df_train[col] = df_train[col].fillna('missing')
            df_pred[col] = df_pred[col].fillna('missing')
        else:
            df_train[col] = df_train[col].fillna(df_train[col].median())
            df_pred[col] = df_pred[col].fillna(df_train[col].median())

cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
if target_col in cat_cols:
    cat_cols.remove(target_col)
if id_col in cat_cols:
    cat_cols.remove(id_col)

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    df_pred[col] = le.transform(df_pred[col].astype(str))
    le_dict[col] = le

num_cols = [c for c in df_train.columns if c not in cat_cols + [target_col, id_col] + coord_cols]

variance_threshold = 0.01
for col in num_cols[:]:
    if df_train[col].var() < variance_threshold:
        num_cols.remove(col)
        df_train.drop(columns=[col], inplace=True)
        df_pred.drop(columns=[col], inplace=True)

corr_matrix = df_train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_train.drop(columns=to_drop, inplace=True)
df_pred.drop(columns=to_drop, inplace=True)
num_cols = [c for c in num_cols if c not in to_drop]

scaler = RobustScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
df_pred[num_cols] = scaler.transform(df_pred[num_cols])

coords_train = df_train[coord_cols].values
coords_pred = df_pred[coord_cols].values

df_train['coord_x'] = coords_train[:, 0]
df_train['coord_y'] = coords_train[:, 1]
df_train['coord_dist_origin'] = np.sqrt(coords_train[:, 0]**2 + coords_train[:, 1]**2)

df_pred['coord_x'] = coords_pred[:, 0]
df_pred['coord_y'] = coords_pred[:, 1]
df_pred['coord_dist_origin'] = np.sqrt(coords_pred[:, 0]**2 + coords_pred[:, 1]**2)

nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coords_train)
distances, indices = nbrs.kneighbors(coords_train)
df_train['nn_mean_dist'] = distances[:, 1:].mean(axis=1)

distances_pred, indices_pred = nbrs.kneighbors(coords_pred)
df_pred['nn_mean_dist'] = distances_pred.mean(axis=1)

feature_cols = [c for c in df_train.columns if c not in [target_col, id_col] + coord_cols]

le_target = LabelEncoder()
y_train = le_target.fit_transform(df_train[target_col])
X_train = df_train[feature_cols].values

class_order = ['Bad', 'Poor', 'Moderate', 'Good', 'High']
class_mapping = {le_target.transform([c])[0]: c for c in class_order}

smote = SMOTE(random_state=42, k_neighbors=5, n_jobs=-1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

class TabularNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        return self.fc4(x)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    cb_params = {
        'iterations': trial.suggest_int('cb_iterations', 500, 2000),
        'depth': trial.suggest_int('cb_depth', 6, 10),
        'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('cb_l2', 1, 10),
        'random_strength': trial.suggest_float('cb_random_strength', 0.5, 2),
        'bagging_temperature': trial.suggest_float('cb_bagging_temp', 0.5, 1.5),
        'task_type': 'GPU',
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 100
    }
    
    nn_lr = trial.suggest_float('nn_lr', 1e-4, 1e-2, log=True)
    nn_epochs = trial.suggest_int('nn_epochs', 50, 200)
    nn_batch = trial.suggest_categorical('nn_batch', [256, 512, 1024])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_bal, y_train_bal)):
        X_tr, X_val = X_train_bal[train_idx], X_train_bal[val_idx]
        y_tr, y_val = y_train_bal[train_idx], y_train_bal[val_idx]
        
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cb_pred = cb_model.predict_proba(X_val)
        
        model = TabularNN(X_tr.shape[1], len(np.unique(y_train_bal))).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=nn_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        train_dataset = TabularDataset(X_tr, y_tr)
        val_dataset = TabularDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=nn_batch, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=nn_batch*2, shuffle=False, num_workers=4, pin_memory=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(nn_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = F.cross_entropy(outputs, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += F.cross_entropy(outputs, y_batch).item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        
        model.eval()
        nn_pred = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                nn_pred.append(F.softmax(outputs, dim=1).cpu().numpy())
        nn_pred = np.vstack(nn_pred)
        
        ensemble_pred = (cb_pred + nn_pred) / 2
        y_pred = np.argmax(ensemble_pred, axis=1)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=False)

best_params = study.best_params

cb_final_params = {
    'iterations': best_params['cb_iterations'],
    'depth': best_params['cb_depth'],
    'learning_rate': best_params['cb_lr'],
    'l2_leaf_reg': best_params['cb_l2'],
    'random_strength': best_params['cb_random_strength'],
    'bagging_temperature': best_params['cb_bagging_temp'],
    'task_type': 'GPU',
    'loss_function': 'MultiClass',
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'verbose': False
}

cb_final = CatBoostClassifier(**cb_final_params)
cb_final.fit(X_train_bal, y_train_bal)

nn_final = TabularNN(X_train_bal.shape[1], len(np.unique(y_train_bal))).to(device)
optimizer = torch.optim.AdamW(nn_final.parameters(), lr=best_params['nn_lr'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

train_dataset_final = TabularDataset(X_train_bal, y_train_bal)
train_loader_final = DataLoader(train_dataset_final, batch_size=best_params['nn_batch'], shuffle=True, num_workers=4, pin_memory=True)

for epoch in range(best_params['nn_epochs']):
    nn_final.train()
    for X_batch, y_batch in train_loader_final:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = nn_final(X_batch)
        loss = F.cross_entropy(outputs, y_batch)
        loss.backward()
        optimizer.step()
    scheduler.step()

nn_final.eval()
X_test_tensor = torch.FloatTensor(X_train).to(device)
with torch.no_grad():
    nn_pred_final = F.softmax(nn_final(X_test_tensor), dim=1).cpu().numpy()

cb_pred_final = cb_final.predict_proba(X_train)
ensemble_pred_final = (cb_pred_final + nn_pred_final) / 2
y_pred_final = np.argmax(ensemble_pred_final, axis=1)

acc_total = accuracy_score(y_train, y_pred_final)
print(f"\nAccuracy Total: {acc_total:.4f}")

print("\nAccuracy por Clase:")
for cls in sorted(np.unique(y_train)):
    mask = y_train == cls
    acc_cls = accuracy_score(y_train[mask], y_pred_final[mask])
    print(f"{class_mapping[cls]}: {acc_cls:.4f}")

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_train, y_pred_final)
print(cm)

X_pred = df_pred[feature_cols].values
X_pred_tensor = torch.FloatTensor(X_pred).to(device)

with torch.no_grad():
    nn_pred_new = F.softmax(nn_final(X_pred_tensor), dim=1).cpu().numpy()

cb_pred_new = cb_final.predict_proba(X_pred)
ensemble_pred_new = (cb_pred_new + nn_pred_new) / 2
y_pred_new = np.argmax(ensemble_pred_new, axis=1)
y_pred_labels = le_target.inverse_transform(y_pred_new)

results_df = pd.DataFrame({
    'SamplingOperations_code': df_pred[id_col].values,
    'IBD_EQR_Status_pred': y_pred_labels
})

results_df.to_csv(output_path, index=False)
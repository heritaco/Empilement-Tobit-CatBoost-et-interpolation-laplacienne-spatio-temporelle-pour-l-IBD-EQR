import pandas as pd
import numpy as np

# Leer el archivo parquet
ruta_archivo = r"C:\Users\japal\Documents\agua\Clasificacion-de-Calidad-del-Agua-IBD-EQR\data\processed\03_CLEAN_COMPLETE_DF_02.parquet"
df = pd.read_parquet(ruta_archivo)

print("=== ANÁLISIS DEL DATASET DE CALIDAD DEL AGUA ===\n")

# Información básica del dataset
print("📊 INFORMACIÓN GENERAL:")
print(f"• Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"• Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"• Valores nulos totales: {df.isnull().sum().sum()}")


# Identificar tipos de variables
variables_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
variables_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
variables_datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
variables_booleanas = df.select_dtypes(include=['bool']).columns.tolist()

print(f"\n🔢 VARIABLES NUMÉRICAS ({len(variables_numericas)}):")
for i, var in enumerate(variables_numericas, 1):
    print(f"  {i:2d}. {var}")

print(f"\n📝 VARIABLES CATEGÓRICAS ({len(variables_categoricas)}):")
for i, var in enumerate(variables_categoricas, 1):
    print(f"  {i:2d}. {var}")
    # Mostrar categorías únicas si son pocas
    unique_vals = df[var].nunique()
    if unique_vals <= 10:
        print(f"      → {unique_vals} categorías: {list(df[var].unique())}")
    else:
        print(f"      → {unique_vals} categorías únicas")

if variables_datetime:
    print(f"\n📅 VARIABLES DE FECHA/HORA ({len(variables_datetime)}):")
    for i, var in enumerate(variables_datetime, 1):
        print(f"  {i:2d}. {var}")

if variables_booleanas:
    print(f"\n✅ VARIABLES BOOLEANAS ({len(variables_booleanas)}):")
    for i, var in enumerate(variables_booleanas, 1):
        print(f"  {i:2d}. {var}")

# Estadísticas descriptivas básicas para variables numéricas
print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS (VARIABLES NUMÉRICAS):")
print(df[variables_numericas].describe().round(2))

# Valores nulos por variable
print(f"\n❌ VALORES NULOS POR VARIABLE:")
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
nulos_df = pd.DataFrame({
    'Variable': nulos.index,
    'Valores_Nulos': nulos.values,
    'Porcentaje': nulos_pct.values
}).sort_values('Valores_Nulos', ascending=False)

print(nulos_df[nulos_df['Valores_Nulos'] > 0].to_string(index=False))

# Resumen final
print(f"\n🏆 RESUMEN DEL DATASET:")
print(f"Este dataset parece ser sobre clasificación de calidad del agua.")
print(f"Contiene {df.shape[0]:,} observaciones con {df.shape[1]} variables en total:")
print(f"  • {len(variables_numericas)} variables numéricas")
print(f"  • {len(variables_categoricas)} variables categóricas") 
if variables_datetime:
    print(f"  • {len(variables_datetime)} variables de fecha/hora")
if variables_booleanas:
    print(f"  • {len(variables_booleanas)} variables booleanas")

# Mostrar las primeras filas
print(f"\n👀 PRIMERAS 5 FILAS DEL DATASET:")
print(df.head())

# Información adicional sobre posibles variables objetivo
print(f"\n🎯 POSIBLES VARIABLES OBJETIVO (basado en nombres):")
posibles_targets = [col for col in df.columns if any(palabra in col.lower() 
                   for palabra in ['calidad', 'quality', 'class', 'target', 'label', 'eqr', 'ibd'])]
if posibles_targets:
    for target in posibles_targets:
        print(f"  • {target}")
        if df[target].dtype in ['object', 'category']:
            print(f"    → Categorías: {df[target].value_counts().to_dict()}")
        else:
            print(f"    → Rango: {df[target].min():.2f} - {df[target].max():.2f}")
else:
    print("  No se identificaron variables objetivo evidentes")
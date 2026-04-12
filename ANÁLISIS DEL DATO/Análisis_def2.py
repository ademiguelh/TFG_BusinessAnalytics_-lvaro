# =====================================
# =====================================
# ANÁLISIS DEL DATO 
# =====================================
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Para facilitar el proceso de análisis, se determinan los paths de los csv y los colores de las 
# criptomonedas para la elaboración de gráficos

PATHS = {
    'BTC': '/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyBitcoin.csv',
    'ETH': '/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyEthereum.csv',
    'SOL': '/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailySolana.csv',
    'XRP': '/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyXRP.csv',
}

COLORS = {
    'BTC': '#F7931A',
    'ETH': '#627EEA',
    'SOL': '#9945FF',
    'XRP': '#00AAE4',
}

# Variables predictoras (X) — ya calculadas en los CSV
FEATURES = ['daily_sentiment_avg', 'sent_lag1', 'sent_lag2', 'sent_lag3',
            'ret_lag1', 'ret_lag2', 'ret_lag3', 'sent_ma7']

# Variable objetivo (Y)
TARGET = 'log_return'

TRAIN_RATIO = 0.80
LAGS        = [0, 1, 2, 3]

# =============================================================================
# CARGA DE DATOS
# =============================================================================
# Cargamos los datos sin dropna para que use todos los dias disponibles, incluso 
# los que no tienen los campos de lag rellenos. El dropna se aplica en la Parte 2 
# antes de entrenar los modelos
 
datos = {}
for cripto, path in PATHS.items():
    df = pd.read_csv(path, parse_dates=['UTC_Time'])
    datos[cripto] = df
    print(f"  {cripto}: {len(df)} días  |  "
          f"{df['UTC_Time'].min().date()} → {df['UTC_Time'].max().date()}")
 
# =============================================================================
# JUSTIFICACIÓN DEL USO DE LAS VARIABLES
# =============================================================================
 
# Correlaciones con lag (Pearson y Spearman) 
 
# Pearson mide la correlación lineal entre sentimiento y precio
# Spearman es más robusta ante valores extremos (outliers)
# Ambas devuelven el coeficiente r y el p-valor

print("\n--- CORRELACIONES SENTIMIENTO vs PRECIO (lag 0-3 días) ---")
print(f"{'Cripto':<6} {'Lag':>4} {'Pearson r':>10} {'p-valor':>9} "
      f"{'Spearman r':>11} {'p-valor':>9} {'Sig':>5}")
print("-" * 60)
 
resultados_corr = []                                                            # Lista vacía donde se irán guardando los resultados
for cripto, df in datos.items():
    for lag in LAGS:
        sent  = df['daily_sentiment_avg'].shift(lag)
        precio = df['price']
        mask  = sent.notna() & precio.notna()
        r_p, p_p = stats.pearsonr(sent[mask], precio[mask])
        r_s, p_s = stats.spearmanr(sent[mask], precio[mask])
        sig = '✓' if p_p < 0.05 else '✗'
        print(f"  {cripto:<4} {lag:>4}    {r_p:>+8.4f}  {p_p:>9.4f}   "
              f"{r_s:>+8.4f}  {p_s:>9.4f}  {sig:>5}")
        resultados_corr.append({
            'Cripto': cripto, 'Lag': lag,
            'Pearson r': round(r_p, 4), 'Pearson p': round(p_p, 4),
            'Spearman r': round(r_s, 4), 'Spearman p': round(p_s, 4),
            'Significativa': 'Sí' if p_p < 0.05 else 'No'
        })
    print()
 
df_corr = pd.DataFrame(resultados_corr)
df_corr.to_csv('resultados_correlaciones.csv', index=False)
print("Tabla guardada en: resultados_correlaciones.csv")
 

# =============================================================================
# VISUALIZACIONES DE LAS VARIABLES
# =============================================================================
 
# Fig 1: Heatmap correlaciones 
 
fig, ax = plt.subplots(figsize=(9, 4))
 
pivot   = df_corr.pivot(index='Cripto', columns='Lag', values='Pearson r')
pivot_p = df_corr.pivot(index='Cripto', columns='Lag', values='Pearson p')
 
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-0.3, vmax=0.3, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(LAGS)))
ax.set_xticklabels([f'Lag {l}' for l in LAGS])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
 
for i in range(len(pivot.index)):
    for j in range(len(LAGS)):
        r_val = pivot.values[i, j]
        p_val = pivot_p.values[i, j]
        txt   = f'{r_val:.3f}{"*" if p_val < 0.05 else ""}'
        ax.text(j, i, txt, ha='center', va='center', fontsize=10)
 
ax.set_title('Correlación de Pearson: sentimiento vs precio\n(* = p < 0.05)',
             fontsize=12)
ax.set_xlabel('Lag (días)')
plt.tight_layout()
plt.savefig('fig01_heatmap_correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig01_heatmap_correlaciones.png")
 
# Fig 2: Series temporales precio + sentimiento 
 
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle('Precio y sentimiento diario por criptomoneda', fontsize=13)
 
for ax, (cripto, df) in zip(axes, datos.items()):
    color = COLORS[cripto]
    ax2   = ax.twinx()
    ax.plot(df['UTC_Time'], df['price'],
            color=color, linewidth=1.5, label='Precio')
    ax.fill_between(df['UTC_Time'], df['price'], alpha=0.07, color=color)
    ax2.plot(df['UTC_Time'], df['daily_sentiment_avg'],
             color='gray', linewidth=0.8, alpha=0.5, label='Sentimiento')
    ax2.plot(df['UTC_Time'], df['sent_ma7'],
             color='black', linewidth=1.2, linestyle='--', label='MA 7d')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax2.set_ylim(-0.7, 0.8)
    ax.set_ylabel('Precio (USD)', color=color)
    ax2.set_ylabel('Sentimiento')
    ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)
 
plt.tight_layout()
plt.savefig('fig02_series_temporales.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig02_series_temporales.png")
 
# Fig 3: Medias móviles del sentimiento 
 
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Medias móviles del sentimiento (7, 14 y 30 días)', fontsize=13)
axes = axes.flatten()
 
for ax, (cripto, df) in zip(axes, datos.items()):
    color = COLORS[cripto]
    ax.bar(df['UTC_Time'], df['daily_sentiment_avg'],
           color=color, alpha=0.15, width=1, label='Sentimiento diario')
    ax.plot(df['UTC_Time'], df['sent_ma7'],
            linestyle='-',  linewidth=1.5, label='MA 7d')
    ax.plot(df['UTC_Time'], df['sent_ma14'],
            linestyle='--', linewidth=1.5, label='MA 14d')
    ax.plot(df['UTC_Time'], df['sent_ma30'],
            linestyle=':',  linewidth=1.5, label='MA 30d')
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel('Sentimiento')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
 
plt.tight_layout()
plt.savefig('fig03_medias_moviles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig03_medias_moviles.png")
 

# =============================================================================
# TEST DE ESTACIONARIEDAD (ADF) Y SELECCIÓN DE ORDEN (p,q) PARA ARIMA/ARIMAX
# =============================================================================

# TEST DE ESTACIONARIEDAD

from statsmodels.tsa.stattools import adfuller

print("\n--- TEST DE ESTACIONARIEDAD (ADF) ---")
print("H0: la serie NO es estacionaria")
print("p < 0.05 → la serie SÍ es estacionaria\n")

for cripto, df in datos.items():
    df_limpio = df[[TARGET]].dropna()
    resultado = adfuller(df_limpio[TARGET], autolag='AIC')
    p_val     = resultado[1]
    est       = "SÍ estacionaria ✓" if p_val < 0.05 else "NO estacionaria ✗"
    print(f"  {cripto}: stat={resultado[0]:.3f}  p={p_val:.4f}  →  {est}")

print()


# GRID SEARCH — SELECCIÓN DE ORDEN (p,q) POR AIC

# Se prueban distintas combinaciones de p y q y se selecciona
# la que produce el menor AIC en los datos de entrenamiento.
# d=0 siempre porque el log-retorno ya es estacionario.

print("--- GRID SEARCH ORDEN (p,q) POR AIC ---\n")

# Se establecen 7 posibles combinaciones para ahorrar tiempo y en base 
# al espectro habitual series económicas

ORDENES_GRID = [(1,0,0), (2,0,0), (0,0,1), (0,0,2),
                (1,0,1), (2,0,1), (1,0,2)]

ordenes_seleccionados = {}

for cripto, df in datos.items():
    df_modelo = df.dropna().reset_index(drop=True)
    serie     = df_modelo[TARGET].values
    exog      = df_modelo['daily_sentiment_avg'].values.reshape(-1, 1)
    n_train   = int(len(df_modelo) * TRAIN_RATIO)
    s_train   = serie[:n_train]
    ex_train  = exog[:n_train]

    # Grid search ARIMA
    mejor_aic_a  = np.inf
    mejor_orden_a = (1, 0, 1)
    for orden in ORDENES_GRID:
        try:
            res = ARIMA(s_train, order=orden).fit(
                method_kwargs={"warn_convergence": False})
            if res.aic < mejor_aic_a:
                mejor_aic_a   = res.aic
                mejor_orden_a = orden
        except Exception:
            pass

    # Grid search ARIMAX
    mejor_aic_ax  = np.inf
    mejor_orden_ax = (1, 0, 1)
    for orden in ORDENES_GRID:
        try:
            res = ARIMA(s_train, exog=ex_train, order=orden).fit(
                method_kwargs={"warn_convergence": False})
            if res.aic < mejor_aic_ax:
                mejor_aic_ax   = res.aic
                mejor_orden_ax = orden
        except Exception:
            pass

    ordenes_seleccionados[cripto] = {
        'arima':  mejor_orden_a,
        'arimax': mejor_orden_ax,
    }

    print(f"  {cripto}:")
    print(f"    ARIMA  mejor orden: {mejor_orden_a}  AIC={mejor_aic_a:.2f}")
    print(f"    ARIMAX mejor orden: {mejor_orden_ax}  AIC={mejor_aic_ax:.2f}")
    print()
 
# =============================================================================
# MODELOS PREDICTIVOS
# =============================================================================
# Aquí se aplica el dropna para que los modelos trabajen solo
# con filas que tengan todas las variables completas.
 
print("\n" + "="*60)
print("PARTE 2 — MODELOS PREDICTIVOS")
print("="*60)
 
def calcular_metricas(real, pred, modelo, cripto):
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae  = mean_absolute_error(real, pred)
    r2   = r2_score(real, pred)
    return {'Cripto': cripto, 'Modelo': modelo,
            'RMSE': round(rmse, 6), 'MAE': round(mae, 6), 'R²': round(r2, 4)}
 
todos_resultados       = []
predicciones_guardadas = {}
 
for cripto, df in datos.items():
 
    # Aplicamos dropna aquí, solo para los modelos
    df_modelo = df.dropna().reset_index(drop=True)
 
    print(f"\n  {cripto} ({len(df_modelo)} filas tras dropna):")
 
    X       = df_modelo[FEATURES].values
    y       = df_modelo[TARGET].values
    n       = len(df_modelo)
    n_train = int(n * TRAIN_RATIO)
 
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    fechas_test     = df_modelo['UTC_Time'].values[n_train:]
 
    # Regresión Lineal
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
 
    rl = LinearRegression()
    rl.fit(X_train_sc, y_train)
    pred_rl = rl.predict(X_test_sc)
 
    met_rl = calcular_metricas(y_test, pred_rl, 'Regresión Lineal', cripto)
    todos_resultados.append(met_rl)
    print(f"    Regresión Lineal  → RMSE={met_rl['RMSE']:.6f}  "
          f"MAE={met_rl['MAE']:.6f}  R²={met_rl['R²']:.4f}")
 
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
 
    met_rf = calcular_metricas(y_test, pred_rf, 'Random Forest', cripto)
    todos_resultados.append(met_rf)
    print(f"    Random Forest     → RMSE={met_rf['RMSE']:.6f}  "
          f"MAE={met_rf['MAE']:.6f}  R²={met_rf['R²']:.4f}")
 
    # ARIMA (sin sentimiento) 
    serie   = df_modelo[TARGET].values
    s_train = serie[:n_train]
    s_test  = serie[n_train:]
 
    pred_arima = []
    hist = list(s_train)
    for t in range(len(s_test)):
        try:
            mod = ARIMA(hist, order=ordenes_seleccionados[cripto]['arima'])
            res = mod.fit(method_kwargs={"warn_convergence": False})
            pred_arima.append(res.forecast(steps=1)[0])
        except Exception:
            pred_arima.append(np.mean(hist))
        hist.append(s_test[t])
 
    pred_arima = np.array(pred_arima)
    met_arima  = calcular_metricas(s_test, pred_arima, 'ARIMA', cripto)
    todos_resultados.append(met_arima)
    print(f"    ARIMA             → RMSE={met_arima['RMSE']:.6f}  "
          f"MAE={met_arima['MAE']:.6f}  R²={met_arima['R²']:.4f}")
 
    # ARIMAX (con sentimiento) 
    exog       = df_modelo['daily_sentiment_avg'].values.reshape(-1, 1)
    exog_train = exog[:n_train]
    exog_test  = exog[n_train:]
 
    pred_arimax = []
    hist_s  = list(s_train)
    hist_ex = list(exog_train.flatten())
    for t in range(len(s_test)):
        try:
            mod = ARIMA(hist_s,
                        exog=np.array(hist_ex).reshape(-1, 1),
                        order=ordenes_seleccionados[cripto]['arimax'])
            res = mod.fit(method_kwargs={"warn_convergence": False})
            pred_arimax.append(
                res.forecast(steps=1,
                             exog=np.array([[exog_test[t, 0]]]))[0])
        except Exception:
            pred_arimax.append(np.mean(hist_s))
        hist_s.append(s_test[t])
        hist_ex.append(exog_test[t, 0])
 
    pred_arimax = np.array(pred_arimax)
    met_arimax  = calcular_metricas(s_test, pred_arimax, 'ARIMAX', cripto)
    todos_resultados.append(met_arimax)
    print(f"    ARIMAX            → RMSE={met_arimax['RMSE']:.6f}  "
          f"MAE={met_arimax['MAE']:.6f}  R²={met_arimax['R²']:.4f}")
 
    mejora = ((met_arima['RMSE'] - met_arimax['RMSE']) / met_arima['RMSE'] * 100)
    print(f"    Mejora ARIMAX vs ARIMA: {mejora:+.2f}%")
 
    predicciones_guardadas[cripto] = {
        'y_test':      y_test,
        'pred_rl':     pred_rl,
        'pred_rf':     pred_rf,
        'pred_arima':  pred_arima,
        'pred_arimax': pred_arimax,
        'fechas_test': fechas_test,
        'rf_model':    rf,
        'rl_coefs':    pd.Series(rl.coef_, index=FEATURES),
    }
 
df_resultados = pd.DataFrame(todos_resultados)
df_resultados.to_csv('resultados_modelos.csv', index=False)
print("\nTabla guardada en: resultados_modelos.csv")
 
print("\n--- MODELO GANADOR POR CRIPTO (menor RMSE) ---")
for cripto in COLORS.keys():
    sub   = df_resultados[df_resultados['Cripto'] == cripto]
    mejor = sub.loc[sub['RMSE'].idxmin()]
    print(f"  {cripto}: {mejor['Modelo']}  "
          f"RMSE={mejor['RMSE']:.6f}  MAE={mejor['MAE']:.6f}  R²={mejor['R²']:.4f}")
 
# =============================================================================
# VISUALIZACIONES DE LOS MODELOS
# =============================================================================
 
 
MODELOS = ['Regresión Lineal', 'Random Forest', 'ARIMA', 'ARIMAX']
KEYS    = ['pred_rl', 'pred_rf', 'pred_arima', 'pred_arimax']
COL_MOD = {
    'Regresión Lineal': '#AAAAAA',          
    'Random Forest':    '#2E5D9E',
    'ARIMA':            '#E08030',
    'ARIMAX':           '#F7931A',
}
 
# Fig 5: Predicciones vs real — un gráfico por modelo 
 
for modelo, key in zip(MODELOS, KEYS):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    nombre_archivo = modelo.lower().replace(' ', '_')
    fig.suptitle(f'{modelo} — Predicción vs real en conjunto test', fontsize=13)
 
    for ax, (cripto, preds) in zip(axes, predicciones_guardadas.items()):
        color  = COLORS[cripto]
        fechas = pd.to_datetime(preds['fechas_test'])
 
        ax.plot(fechas, preds['y_test'],
                color='black', linewidth=1.2, alpha=0.8, label='Real')
        ax.plot(fechas, preds[key],
                color=color, linewidth=1.2, linestyle='--',
                alpha=0.85, label=modelo)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
 
        met = df_resultados[(df_resultados['Cripto'] == cripto) &
                             (df_resultados['Modelo'] == modelo)].iloc[0]
        ax.text(0.01, 0.95,
                f"RMSE={met['RMSE']:.5f}  MAE={met['MAE']:.5f}  R²={met['R²']:.4f}",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
 
        ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
        ax.set_ylabel('Log-retorno')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
 
    plt.tight_layout()
    plt.savefig(f'fig05_{nombre_archivo}_predicciones.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig05_{nombre_archivo}_predicciones.png")
 
# Fig 6: Comparación de métricas — todos los modelos juntos 
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Comparación de métricas entre modelos', fontsize=13)
 
criptos = list(COLORS.keys())
x       = np.arange(len(criptos))
ancho   = 0.20
 
for ax, metrica in zip(axes, ['RMSE', 'MAE', 'R²']):
    for i, modelo in enumerate(MODELOS):
        vals   = [df_resultados[(df_resultados['Cripto'] == c) &
                                (df_resultados['Modelo'] == modelo)][metrica].values[0]
                  for c in criptos]
        offset = (i - 1.5) * ancho
        ax.bar(x + offset, vals, ancho, label=modelo,
               color=COL_MOD[modelo], edgecolor='white', alpha=0.88)
 
    ax.set_xticks(x)
    ax.set_xticklabels(criptos)
    ax.set_title(metrica, fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylabel('Menor = mejor' if metrica != 'R²' else 'Mayor = mejor')
 
plt.tight_layout()
plt.savefig('fig06_comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig06_comparacion_metricas.png")
 
# Fig 7: ARIMA vs ARIMAX 
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ARIMA vs ARIMAX — efecto del sentimiento en la predicción', fontsize=13)
axes = axes.flatten()
 
for ax, (cripto, preds) in zip(axes, predicciones_guardadas.items()):
    color  = COLORS[cripto]
    fechas = pd.to_datetime(preds['fechas_test'])
 
    ax.plot(fechas, preds['y_test'],
            color='black', linewidth=1.2, alpha=0.8, label='Real')
    ax.plot(fechas, preds['pred_arima'],
            color='gray', linewidth=1, linestyle='--', alpha=0.7, label='ARIMA')
    ax.plot(fechas, preds['pred_arimax'],
            color=color, linewidth=1.2, label='ARIMAX')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
 
    met_a  = df_resultados[(df_resultados['Cripto'] == cripto) &
                            (df_resultados['Modelo'] == 'ARIMA')].iloc[0]
    met_ax = df_resultados[(df_resultados['Cripto'] == cripto) &
                            (df_resultados['Modelo'] == 'ARIMAX')].iloc[0]
    mejora = (met_a['RMSE'] - met_ax['RMSE']) / met_a['RMSE'] * 100
 
    ax.text(0.01, 0.95,
            f"ARIMA   RMSE={met_a['RMSE']:.5f}\n"
            f"ARIMAX RMSE={met_ax['RMSE']:.5f}\n"
            f"Mejora: {mejora:+.2f}%",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
 
    ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel('Log-retorno')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
 
plt.tight_layout()
plt.savefig('fig07_arima_vs_arimax.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig07_arima_vs_arimax.png")
 
#  Fig 8: Importancia de variables — Random Forest 
 
leyenda_colores = [Patch(color='#F7931A', label='Variable de sentimiento'),
                   Patch(color='#AAAAAA', label='Variable de precio')]
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Importancia de variables — Random Forest', fontsize=13)
axes = axes.flatten()
 
for ax, (cripto, preds) in zip(axes, predicciones_guardadas.items()):
    color = COLORS[cripto]
    rf    = preds['rf_model']
    imp   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    cols  = ['#F7931A' if 'sent' in f else '#AAAAAA' for f in imp.index]
 
    ax.barh(imp.index, imp.values, color=cols, edgecolor='white', height=0.6)
    for i, (feat, val) in enumerate(imp.items()):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=8)
 
    ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('Importancia')
    ax.grid(axis='x', alpha=0.2)
 
fig.legend(handles=leyenda_colores, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig('fig08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig08_feature_importance.png")
 
#  Fig 9: Coeficientes Regresión Lineal ─
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Coeficientes estandarizados — Regresión Lineal', fontsize=13)
axes = axes.flatten()
 
for ax, (cripto, preds) in zip(axes, predicciones_guardadas.items()):
    color = COLORS[cripto]
    coefs = preds['rl_coefs'].sort_values(key=abs, ascending=True)
    cols  = ['#F7931A' if 'sent' in f else '#AAAAAA' for f in coefs.index]
 
    ax.barh(coefs.index, coefs.values, color=cols, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(cripto, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('Coeficiente estandarizado')
    ax.grid(axis='x', alpha=0.2)
 
fig.legend(handles=leyenda_colores, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig('fig09_coeficientes_rl.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig09_coeficientes_rl.png")
 
# Fig 10: Tabla resumen final 
 
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
 
filas = []
for cripto in COLORS.keys():
    fila = [cripto]
    for modelo in MODELOS:
        sub  = df_resultados[(df_resultados['Cripto'] == cripto) &
                              (df_resultados['Modelo'] == modelo)]
        fila += [f"{sub['RMSE'].values[0]:.5f}",
                 f"{sub['MAE'].values[0]:.5f}",
                 f"{sub['R²'].values[0]:.4f}"]
    filas.append(fila)
 
col_labels = ['Cripto']
for modelo in MODELOS:
    col_labels += [f'{modelo}\nRMSE', f'{modelo}\nMAE', f'{modelo}\nR²']
 
t = ax.table(cellText=filas, colLabels=col_labels,
             cellLoc='center', loc='center')
t.auto_set_font_size(False)
t.set_fontsize(7.5)
t.scale(1.1, 1.9)
 
for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1F3864')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F5F7FA')
 
ax.set_title('Resumen de métricas por modelo y criptomoneda',
             fontsize=12, pad=20)
plt.tight_layout()
plt.savefig('fig10_tabla_resumen.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig10_tabla_resumen.png")
 
# =============================================================================
# RESUMEN FINAL
# =============================================================================
 
for cripto in COLORS.keys():
    sub   = df_resultados[df_resultados['Cripto'] == cripto]
    mejor = sub.loc[sub['RMSE'].idxmin()]
    print(f"  {cripto}: {mejor['Modelo']}  "
          f"RMSE={mejor['RMSE']:.6f}  MAE={mejor['MAE']:.6f}  R²={mejor['R²']:.4f}")
 

 

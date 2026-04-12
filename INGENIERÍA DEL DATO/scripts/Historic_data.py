
# =====================================
# =====================================
# INGENIERÍA DEL DATO 1/3
# =====================================
# =====================================

# =====================================
# PRUEBA PARA VERIFICAR QUE FUNCIONA VSCODE
# =====================================

print("helloworld")

# =====================================
# IMPORTAR LIBRERIAS
# =====================================

import pandas as pd

# =====================================
# AÑADIR COLUMNA "crypto" AL DF
# =====================================

file_path = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/btc-usd-max.csv"
file_path2 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/eth-usd-max.csv"
file_path3 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/sol-usd-max.csv"
file_path4 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/xrp-usd-max.csv"
btc = pd.read_csv(file_path)
eth = pd.read_csv(file_path2)
sol = pd.read_csv(file_path3)
xrp = pd.read_csv(file_path4)
print(btc.head())
btc ["crypto"] = "btc"
eth ["crypto"] = "eth"
sol ["crypto"] = "sol"
xrp ["crypto"] = "xrp"
print(btc.head())

# =====================================
# COMPRUEBO DUPLICADOS
# =====================================

print(f"Total de filas duplicadas bitcoin: {btc.duplicated().sum()}")
print(f"Total de filas duplicadas ethereum: {eth.duplicated().sum()}")
print(f"Total de filas duplicadas solana: {sol.duplicated().sum()}")
print(f"Total de filas duplicadas xrp: {xrp.duplicated().sum()}")

# =====================================
# COMPROBACIÓN DE VALORES NULOS EN price
# =====================================

# 1º comprobamos nulos
print(f"Nulos en el precio histórico (Bitcoin): {btc['price'].isna().sum()}")
print(f"Nulos en el precio histórico (Ethereum): {eth['price'].isna().sum()}")
print(f"Nulos en el precio histórico (Solana): {sol['price'].isna().sum()}")
print(f"Nulos en el precio histórico (XRP): {xrp['price'].isna().sum()}")


# =====================================
# CAMBIAR LA FECHA
# =====================================

# 1º convertimoas a datatime

btc["snapped_at"] = pd.to_datetime(btc["snapped_at"], utc=True)
eth["snapped_at"] = pd.to_datetime(eth["snapped_at"], utc=True)
sol["snapped_at"] = pd.to_datetime(sol["snapped_at"], utc=True)
xrp["snapped_at"] = pd.to_datetime(xrp["snapped_at"], utc=True)

# 2º nos quedamos con la fecha unicamente

btc["snapped_at"] = btc["snapped_at"].dt.date
eth["snapped_at"] = eth["snapped_at"].dt.date
sol["snapped_at"] = sol["snapped_at"].dt.date
xrp["snapped_at"] = xrp["snapped_at"].dt.date

# 3º compruebo que la fecha esté bien

print(btc.head())

# 4º Restar un dia a toda la columna de snapped at (hacemos esto porque es el precio a dia x a las 00:00, nosotros queremos
# que sea el precio de cierre Del dia antes

btc["snapped_at"] = btc["snapped_at"] - pd.Timedelta(days=1)
eth["snapped_at"] = eth["snapped_at"] - pd.Timedelta(days=1)
sol["snapped_at"] = sol["snapped_at"] - pd.Timedelta(days=1)
xrp["snapped_at"] = xrp["snapped_at"] - pd.Timedelta(days=1)
print(btc.head())

#Cambiar el nombre de la columna de tiempo

btc = btc.rename(columns={"snapped_at": "UTC_Time"})
eth = eth.rename(columns={"snapped_at": "UTC_Time"})
sol = sol.rename(columns={"snapped_at": "UTC_Time"})
xrp = xrp.rename(columns={"snapped_at": "UTC_Time"})
print(btc.head())

# =====================================
# COMPRUEBO DUPLICADOS
# =====================================

print(f"Total de filas duplicadas bitcoin: {btc.duplicated().sum()}")
print(f"Total de filas duplicadas ethereum: {eth.duplicated().sum()}")
print(f"Total de filas duplicadas solana: {sol.duplicated().sum()}")
print(f"Total de filas duplicadas xrp: {xrp.duplicated().sum()}")

#como no hay filas duplicadas, no hay que hacer nada. Si no, tendría que eliminarlas.

# =====================================
# ME GUARDO EL DF COMO CSV PARA HACER EL JOIN
# =====================================

btc.to_csv("PRECIOS DIARIOS BTC.csv", index=False)
eth.to_csv("PRECIOS DIARIOS ETH.csv", index=False)
sol.to_csv("PRECIOS DIARIOS SOL.csv", index=False)
xrp.to_csv("PRECIOS DIARIOS XRP.csv", index=False)

# =====================================
# =====================================
# INGENIERÍA DEL DATO 2/3
# =====================================
# =====================================

# =====================================
# IMPORTAR LIBRERIAS
# =====================================

import pandas as pd
import numpy as np
import re as re 

import emoji as emo
import nltk
import vaderSentiment

from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====================================
# AÑADIR COLUMNA "crypto" AL DF
# =====================================

file_path1 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/btc_tuits.csv"
file_path2 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/eth_tuits.csv"
file_path3 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/solana_tuits.csv"
file_path4 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Raw_Data/xrp_tuits.csv"
dfbtc = pd.read_csv(file_path1)
dfeth = pd.read_csv(file_path2)
dfsol = pd.read_csv(file_path3)
dfxrp = pd.read_csv(file_path4)
print(dfbtc.head())
dfbtc ["crypto"] = "btc"
dfeth ["crypto"] = "eth"
dfsol ["crypto"] = "sol"
dfxrp ["crypto"] = "xrp"
print(dfbtc.head())

# =====================================
# ELIMINAR LAS COLUMNAS QUE NO NOS SIRVEN Y ALMACENAR EN DF2
# =====================================

print(dfbtc.columns)
dfbtc2 = dfbtc.drop(columns=["Query_Str", "Post_URL", "Author_Name", "Author_Web_Page_URL", "Verified_Status", "Ads", "Post_ID", "Tweet_URL",
                       "Tweet_Image_URL", "Reply_to_Whom", "Reply_to_Whom_URL", "Reply_to_Whom_Username", "Reply_to_Whom_Handle",
                       "Language", "Type"])
dfeth2 = dfeth.drop(columns=["Query_Str", "Post_URL", "Author_Name", "Author_Web_Page_URL", "Verified_Status", "Ads", "Post_ID", "Tweet_URL",
                       "Tweet_Image_URL", "Reply_to_Whom", "Reply_to_Whom_URL", "Reply_to_Whom_Username", "Reply_to_Whom_Handle",
                       "Language", "Type"])
dfsol2 = dfsol.drop(columns=["Query_Str", "Post_URL", "Author_Name", "Author_Web_Page_URL", "Verified_Status", "Ads", "Post_ID", "Tweet_URL",
                       "Tweet_Image_URL", "Reply_to_Whom", "Reply_to_Whom_URL", "Reply_to_Whom_Username", "Reply_to_Whom_Handle",
                       "Language", "Type"])
dfxrp2 = dfxrp.drop(columns=["Query_Str", "Post_URL", "Author_Name", "Author_Web_Page_URL", "Verified_Status", "Ads", "Post_ID", "Tweet_URL",
                       "Tweet_Image_URL", "Reply_to_Whom", "Reply_to_Whom_URL", "Reply_to_Whom_Username", "Reply_to_Whom_Handle",
                       "Language", "Type"])
print(dfbtc2.columns)
print(dfbtc2.head())
dfbtc2.head()

# =====================================
# COMPROBAR SI HAY DUPLICADOS TRAS ELIMINAR LAS VARIABLES QUE NO NOS SIRVEN
# =====================================

print(f"Total de filas duplicadas bitcoin: {dfbtc2.duplicated().sum()}")
print(f"Total de filas duplicadas ethereum: {dfeth2.duplicated().sum()}")
print(f"Total de filas duplicadas solana: {dfsol2.duplicated().sum()}")
print(f"Total de filas duplicadas xrp: {dfxrp2.duplicated().sum()}")

#como no hay filas duplicadas, no hay que hacer nada. Si no, tendría que eliminarlas.

# =====================================
# COMPROBACIÓN DE VALORES NULOS O VACÍOS EN Tweet_Content
# =====================================

# 1º comprobamos nulos
print(f"Nulos en Tweet_Content (Bitcoin): {dfbtc2['Tweet_Content'].isna().sum()}")
print(f"Nulos en Tweet_Content (Ethereum): {dfeth2['Tweet_Content'].isna().sum()}")
print(f"Nulos en Tweet_Content (Solana): {dfsol2['Tweet_Content'].isna().sum()}")
print(f"Nulos en Tweet_Content (XRP): {dfxrp2['Tweet_Content'].isna().sum()}")

# 2º comprobamos vacíos
print(f"\nVacíos en Tweet_Content (Bitcoin): {(dfbtc2['Tweet_Content'].str.strip() == '').sum()}")
print(f"Vacíos en Tweet_Content (Ethereum): {(dfeth2['Tweet_Content'].str.strip() == '').sum()}")
print(f"Vacíos en Tweet_Content (Solana): {(dfsol2['Tweet_Content'].str.strip() == '').sum()}")
print(f"Vacíos en Tweet_Content (XRP): {(dfxrp2['Tweet_Content'].str.strip() == '').sum()}")

# Solo hay un nulo en xrp Eliminamos la fila con nulo en Tweet_Content (XRP)
dfxrp2 = dfxrp2.dropna(subset=["Tweet_Content"])
print(f"Filas XRP tras eliminar nulo: {len(dfxrp2)}")

# =====================================
# MODIFICAR LA COLUMNA DE UTC_Time PARA QUE SOLO SALGA LA FECHA
# =====================================

# 1º convertimos a datatime

dfbtc2["UTC_Time"] = pd.to_datetime(dfbtc2["UTC_Time"], utc=True)
dfeth2["UTC_Time"] = pd.to_datetime(dfeth2["UTC_Time"], utc=True)
dfsol2["UTC_Time"] = pd.to_datetime(dfsol2["UTC_Time"], utc=True)
dfxrp2["UTC_Time"] = pd.to_datetime(dfxrp2["UTC_Time"], utc=True)

# 2º nos quedamos con la fecha unicamente

dfbtc2["UTC_Time"] = dfbtc2["UTC_Time"].dt.date
dfeth2["UTC_Time"] = dfeth2["UTC_Time"].dt.date
dfsol2["UTC_Time"] = dfsol2["UTC_Time"].dt.date
dfxrp2["UTC_Time"] = dfxrp2["UTC_Time"].dt.date

# 3º compruebo que la fecha esté bien

print(dfbtc2["UTC_Time"].head())

# =====================================
# AÑADIMOS PALABRAS AL DICCIONARIO DE SENTIMIENTO QUE TIENEN
# QUE VER CON CRYPTO (CHATGPT)
# =====================================

analyzer = SentimentIntensityAnalyzer()    # Vader tiene un diccionario interno que se llama analyzer.lexicon y se puede actualizar usando una escala de -4 a 4

analyzer.lexicon.update({
    "moon": 3.0,
    "bullish": 2.5,
    "bearish": -2.5,
    "rekt": -3.0,
    "rugpull": -4.0,
    "fomo": 2.0,
    "hodl": 1.5,
    "bitcoin": 1.0,
    "btc": 1.0,
    "crypto": 1.0,
    "etf": 2.0,
    "spot": 0.5,
    "breaking": 0.0,
    "market": 0.5,
    "money": 0.5,
    "blackrock": 2.0,
    "sec": -1.5,
    "price": 0.0,
    "next": 0.0,
    "world": 0.0,
    "buy": 2.5,
    "bull": 3.5,
    "bullish": 3.5,
    "bear": -3.5,
    "bearish": -3.5,
    "bank": -0.5,
    "halving": 3.0,
    "million": 0.0,
    "mining": 1.0,
    "asset": 1.0,
    "fund": 1.0,
    "institutional": 2.5,
    "approval": 2.5,
    "approved": 3.0,
    "launch": 1.5,
    "listing": 2.0,
    "adoption": 3.0,
    "accumulation": 3.0,
    "whale": 1.5,
    "pump": 3.0,
    "dump": -3.0,
    "crash": -4.0,
    "collapse": -4.0,
    "ban": -3.5,
    "regulation": -1.5,
    "lawsuit": -2.5,
    "fraud": -4.0,
    "scam": -4.0,
    "rug": -4.0,
    "rugpull": -4.0,
    "hack": -3.5,
    "liquidation": -3.0,
    "sell": -2.5,
    "resistance": -1.5,
    "support": 1.5,
    "breakout": 3.0,
    "rally": 3.5,
    "surge": 3.5,
    "ath": 4.0,
    "alltimehigh": 4.0,
    "high": 1.0,
    "low": -1.5,
    "drop": -3.0,
    "fall": -3.0,
    "volatility": -1.0,
    "risk": -2.0,
    "fear": -3.0,
    "greed": 2.0,
    "fomo": 2.5,
    "hodl": 2.5,
    "long": 2.0,
    "short": -2.0,
    "leverage": -1.5,
    "margin": -1.0,
    "profit": 3.0,
    "loss": -3.0,
    "gain": 3.0,
    "bearmarket": -4.0,
    "bullmarket": 4.0,
    "trend": 1.0,
    "momentum": 2.0,
    "growth": 3.0,
    "innovation": 2.5,
    "technology": 1.5,
    "blockchain": 1.5,
    "decentralized": 2.0,
    "defi": 2.0,
    "staking": 2.0,
    "yield": 1.5,
    "airdrop": 2.0,
    "token": 0.5,
    "altcoin": 0.5,
    "portfolio": 0.5,
    "investment": 2.0,
    "inflow": 3.0,
    "outflow": -3.0,
    "demand": 2.5,
    "supply": -1.0,
    "recession": -3.5,
    "inflation": -2.5,
    "ratehike": -3.5,
    "cut": 2.0,
    "approvaldelay": -2.5,
    "etfapproval": 4.0,
    "bankruptcy": -4.0,
    "scam": -3.5
    })


# =====================================
# ELIMINAMOS LAS STOPWORDS Y HACEMOS LA TOKENIZACIÓN
# =====================================

stop_words = set(stopwords.words("english"))
analyzer = SentimentIntensityAnalyzer()

def clean_text(s: str) -> str:
    s = str(s)
    s = emo.replace_emoji(s, replace="")              # quita emojis
    s = re.sub(r"http\S+|www\.\S+", "", s)              # quita URLs
    s = re.sub(r"@\w+", "", s)                          # quita menciones
    s = re.sub(r"#", "", s)                             # quita # (mantén la palabra)
    s = re.sub(r"[^A-Za-z\s]", " ", s)                  # deja letras y espacios
    s = s.lower()
    tokens = [w for w in s.split() if w not in stop_words]
    return " ".join(tokens)

def vader_score(s: str) -> float:
    return analyzer.polarity_scores(s)["compound"]      # -1 a 1

dfbtc2["text_clean"] = dfbtc2["Tweet_Content"].fillna("").apply(clean_text)
dfbtc2["sentiment_score"] = dfbtc2["text_clean"].apply(vader_score)

dfeth2["text_clean"] = dfeth2["Tweet_Content"].fillna("").apply(clean_text)
dfeth2["sentiment_score"] = dfeth2["text_clean"].apply(vader_score)

dfsol2["text_clean"] = dfsol2["Tweet_Content"].fillna("").apply(clean_text)
dfsol2["sentiment_score"] = dfsol2["text_clean"].apply(vader_score)

dfxrp2["text_clean"] = dfxrp2["Tweet_Content"].fillna("").apply(clean_text)
dfxrp2["sentiment_score"] = dfxrp2["text_clean"].apply(vader_score)

dfbtc2["sentiment_score"] = (
    dfbtc2["Tweet_Content"]
    .fillna("")
    .apply(clean_text)
    .apply(vader_score)
)

dfeth2["sentiment_score"] = (
    dfeth2["Tweet_Content"]
    .fillna("")
    .apply(clean_text)
    .apply(vader_score)
)

dfsol2["sentiment_score"] = (
    dfsol2["Tweet_Content"]
    .fillna("")
    .apply(clean_text)
    .apply(vader_score)
)

dfxrp2["sentiment_score"] = (
    dfxrp2["Tweet_Content"]
    .fillna("")
    .apply(clean_text)
    .apply(vader_score)
)

print(dfbtc2.head())

# =====================================
# CREAR UNA COLUMNA CON LOS PESOS ALMACENADOS
# =====================================

dfbtc2["daily_sentiment_avg"] = (
    dfbtc2
    .groupby("UTC_Time")["sentiment_score"]
    .transform("mean")
)

dfeth2["daily_sentiment_avg"] = (
    dfeth2
    .groupby("UTC_Time")["sentiment_score"]
    .transform("mean")
)

dfsol2["daily_sentiment_avg"] = (
    dfsol2
    .groupby("UTC_Time")["sentiment_score"]
    .transform("mean")
)

dfxrp2["daily_sentiment_avg"] = (
    dfxrp2
    .groupby("UTC_Time")["sentiment_score"]
    .transform("mean")
)
print(dfbtc2.head())

# =====================================
# ADICIONALMENTE VOY A CREAR OTRO DF DIARIO (SOLO DE PRUEBA)
# =====================================

df_daily = (
    dfbtc2
    .groupby("UTC_Time")
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        tweet_count=("sentiment_score", "count")
    )
    .reset_index()
)
print(df_daily.head())

df_daily = (
    dfeth2
    .groupby("UTC_Time")
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        tweet_count=("sentiment_score", "count")
    )
    .reset_index()
)
print(df_daily.head())

df_daily = (
    dfsol2
    .groupby("UTC_Time")
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        tweet_count=("sentiment_score", "count")
    )
    .reset_index()
)
print(df_daily.head())

df_daily = (
    dfxrp2
    .groupby("UTC_Time")
    .agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        tweet_count=("sentiment_score", "count")
    )
    .reset_index()
)
print(df_daily.head())

# =====================================
# HAGO LA JOIN ENTRE ESTE DF Y EL DE LOS PRECIOS DIARIOS
# =====================================

#1º cargo el df de los precios diarios de btc

dfpricebtc = pd.read_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Precios diarios limpios/PRECIOS DIARIOS BTC.csv")

dfpriceeth = pd.read_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Precios diarios limpios/PRECIOS DIARIOS ETH.csv")

dfpricesol = pd.read_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Precios diarios limpios/PRECIOS DIARIOS SOL.csv")

dfpricexrp = pd.read_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/Precios diarios limpios/PRECIOS DIARIOS XRP.csv")

#2º aseguro que las fechas y las cryptos coincidan

dfpricebtc["UTC_Time"] = pd.to_datetime(dfpricebtc["UTC_Time"])
dfbtc2["UTC_Time"] = pd.to_datetime(dfbtc2["UTC_Time"])

dfpricebtc["crypto"] = dfpricebtc["crypto"].str.lower()
dfbtc2["crypto"] = dfbtc2["crypto"].str.lower()

dfpriceeth["UTC_Time"] = pd.to_datetime(dfpriceeth["UTC_Time"])
dfeth2["UTC_Time"] = pd.to_datetime(dfeth2["UTC_Time"])

dfpriceeth["crypto"] = dfpriceeth["crypto"].str.lower()
dfeth2["crypto"] = dfeth2["crypto"].str.lower()

dfpricesol["UTC_Time"] = pd.to_datetime(dfpricesol["UTC_Time"])
dfsol2["UTC_Time"] = pd.to_datetime(dfsol2["UTC_Time"])

dfpricesol["crypto"] = dfpricesol["crypto"].str.lower()
dfsol2["crypto"] = dfsol2["crypto"].str.lower()

dfpricexrp["UTC_Time"] = pd.to_datetime(dfpricexrp["UTC_Time"])
dfxrp2["UTC_Time"] = pd.to_datetime(dfxrp2["UTC_Time"])

dfpricexrp["crypto"] = dfpricexrp["crypto"].str.lower()
dfxrp2["crypto"] = dfxrp2["crypto"].str.lower()

#3º hacemos la join (como si fuese SQL) en python

df_Fbitcoin = pd.merge(
    dfbtc2,
    dfpricebtc[["UTC_Time", "crypto", "price"]],
    on=["UTC_Time", "crypto"],
    how="left"
)

df_Fethereum = pd.merge(
    dfeth2,
    dfpriceeth[["UTC_Time", "crypto", "price"]],
    on=["UTC_Time", "crypto"],
    how="left"
)

df_Fsolana = pd.merge(
    dfsol2,
    dfpricesol[["UTC_Time", "crypto", "price"]],
    on=["UTC_Time", "crypto"],
    how="left"
)

df_Fxrp = pd.merge(
    dfxrp2,
    dfpricexrp[["UTC_Time", "crypto", "price"]],
    on=["UTC_Time", "crypto"],
    how="left"
)

print("Así queda el df de bitcoin limpio")
print(df_Fbitcoin.head())

print("Así queda el df de ethereum limpio")
print(df_Fethereum.head())

print("Así queda el df de solana limpio")
print(df_Fsolana.head())

print("Así queda el df de xrp limpio")
print(df_Fxrp.head())

print (df_Fbitcoin.columns)
# =====================================
# REVISO SI HAY DUPLICADOS DESPUÉS DE LA JOIN
# =====================================

print(f"Total de filas duplicadas bitcoin: {df_Fbitcoin.duplicated().sum()}")
print(f"Total de filas duplicadas ethereum: {df_Fethereum.duplicated().sum()}")
print(f"Total de filas duplicadas solana: {df_Fsolana.duplicated().sum()}")
print(f"Total de filas duplicadas xrp: {df_Fxrp.duplicated().sum()}")

#como no hay filas duplicadas, no hay que hacer nada. Si no, tendría que eliminarlas.

# =====================================
# CREO OTRO DF QUE COMPILE UNICAMENTE VALORES DIARIOS (FECHA, PRECIO, SENTIMIENTO DIARIO)
# =====================================

df_daily_btc = df_Fbitcoin.groupby("UTC_Time").agg(
    price=("price", "first"),
    daily_sentiment_avg=("daily_sentiment_avg", "first")
).reset_index()

df_daily_eth = df_Fethereum.groupby("UTC_Time").agg(
    price=("price", "first"),
    daily_sentiment_avg=("daily_sentiment_avg", "first")
).reset_index()

df_daily_sol = df_Fsolana.groupby("UTC_Time").agg(
    price=("price", "first"),
    daily_sentiment_avg=("daily_sentiment_avg", "first")
).reset_index()

df_daily_xrp = df_Fxrp.groupby("UTC_Time").agg(
    price=("price", "first"),
    daily_sentiment_avg=("daily_sentiment_avg", "first")
).reset_index()

# =====================================
# CALCULO EL LOG_RETURN Y LO ALMACENO EN EL DF "df_daily_xxx"(DF QUE SE USARÁ PARA LA MODELIZACIÓN)
# =====================================


df_daily_btc["log_return"] = np.log(df_daily_btc["price"]).diff()
df_daily_eth["log_return"] = np.log(df_daily_eth["price"]).diff()
df_daily_sol["log_return"] = np.log(df_daily_sol["price"]).diff()
df_daily_xrp["log_return"] = np.log(df_daily_xrp["price"]).diff()

# El log_return será la variable que intentaré predecir con los moselos, que es una variación porcentual

# =====================================
# CALCULO LOS LAGS DE PRECIO Y LOS ALMACENO EN EL DF "df_daily_xxx"(DF QUE SE USARÁ PARA LA MODELIZACIÓN)
# =====================================

df_daily_btc["ret_lag1"] = df_daily_btc["log_return"].shift(1)
df_daily_btc["ret_lag2"] = df_daily_btc["log_return"].shift(2)
df_daily_btc["ret_lag3"] = df_daily_btc["log_return"].shift(3)

df_daily_eth["ret_lag1"] = df_daily_eth["log_return"].shift(1)
df_daily_eth["ret_lag2"] = df_daily_eth["log_return"].shift(2)
df_daily_eth["ret_lag3"] = df_daily_eth["log_return"].shift(3)

df_daily_sol["ret_lag1"] = df_daily_sol["log_return"].shift(1)
df_daily_sol["ret_lag2"] = df_daily_sol["log_return"].shift(2)
df_daily_sol["ret_lag3"] = df_daily_sol["log_return"].shift(3)

df_daily_xrp["ret_lag1"] = df_daily_xrp["log_return"].shift(1)
df_daily_xrp["ret_lag2"] = df_daily_xrp["log_return"].shift(2)
df_daily_xrp["ret_lag3"] = df_daily_xrp["log_return"].shift(3)

# =====================================
# CALCULO LOS LAGS DE SENTIMIENTO Y LOS ALMACENO EN EL DF "df_daily_xxx"(DF QUE SE USARÁ PARA LA MODELIZACIÓN)
# =====================================

df_daily_btc["sent_lag1"] = df_daily_btc["daily_sentiment_avg"].shift(1)
df_daily_btc["sent_lag2"] = df_daily_btc["daily_sentiment_avg"].shift(2)
df_daily_btc["sent_lag3"] = df_daily_btc["daily_sentiment_avg"].shift(3)

df_daily_eth["sent_lag1"] = df_daily_eth["daily_sentiment_avg"].shift(1)
df_daily_eth["sent_lag2"] = df_daily_eth["daily_sentiment_avg"].shift(2)
df_daily_eth["sent_lag3"] = df_daily_eth["daily_sentiment_avg"].shift(3)

df_daily_sol["sent_lag1"] = df_daily_sol["daily_sentiment_avg"].shift(1)
df_daily_sol["sent_lag2"] = df_daily_sol["daily_sentiment_avg"].shift(2)
df_daily_sol["sent_lag3"] = df_daily_sol["daily_sentiment_avg"].shift(3)

df_daily_xrp["sent_lag1"] = df_daily_xrp["daily_sentiment_avg"].shift(1)
df_daily_xrp["sent_lag2"] = df_daily_xrp["daily_sentiment_avg"].shift(2)
df_daily_xrp["sent_lag3"] = df_daily_xrp["daily_sentiment_avg"].shift(3)

# =====================================
# CALCULO LA MEDIA MÓVIL DEL SENTIMIENTO (7 DÍAS) Y LA ALMACENO EN EL DF "df_daily_xxx"
# =====================================

df_daily_btc["sent_ma7"] = df_daily_btc["daily_sentiment_avg"].rolling(7).mean()
df_daily_eth["sent_ma7"] = df_daily_eth["daily_sentiment_avg"].rolling(7).mean()
df_daily_sol["sent_ma7"] = df_daily_sol["daily_sentiment_avg"].rolling(7).mean()
df_daily_xrp["sent_ma7"] = df_daily_xrp["daily_sentiment_avg"].rolling(7).mean()

df_daily_btc["sent_ma14"] = df_daily_btc["daily_sentiment_avg"].rolling(14).mean()
df_daily_eth["sent_ma14"] = df_daily_eth["daily_sentiment_avg"].rolling(14).mean()
df_daily_sol["sent_ma14"] = df_daily_sol["daily_sentiment_avg"].rolling(14).mean()
df_daily_xrp["sent_ma14"] = df_daily_xrp["daily_sentiment_avg"].rolling(14).mean()

df_daily_btc["sent_ma30"] = df_daily_btc["daily_sentiment_avg"].rolling(30).mean()
df_daily_eth["sent_ma30"] = df_daily_eth["daily_sentiment_avg"].rolling(30).mean()
df_daily_sol["sent_ma30"] = df_daily_sol["daily_sentiment_avg"].rolling(30).mean()
df_daily_xrp["sent_ma30"] = df_daily_xrp["daily_sentiment_avg"].rolling(30).mean()

# =====================================
# DESCARGO LOS DFs para su posterior análisis
# =====================================

df_Fbitcoin.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalBitcoin.csv", index=False)
df_Fethereum.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalEthereum.csv", index=False)
df_Fsolana.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalSolana.csv", index=False)
df_Fxrp.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalXRP.csv", index=False)

df_daily_btc.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyBitcoin.csv", index=False)
df_daily_eth.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyEthereum.csv", index=False)
df_daily_sol.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailySolana.csv", index=False)
df_daily_xrp.to_csv("/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinaldailyXRP.csv", index=False)
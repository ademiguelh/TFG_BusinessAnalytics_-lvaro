
# =====================================
# =====================================
# INGENIERÍA DEL DATO 3/3
# =====================================
# =====================================

# =====================================
# IMPORTAR LIBRERIAS
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# CARGAMOS LOS DF
# =====================================

file_path1 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalBitcoin.csv"
file_path2 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalEthereum.csv"
file_path3 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalSolana.csv"
file_path4 = "/Users/alvarodemiguel/TFG/INGENIERÍA DEL DATO/dfLimpios/DFfinalXRP.csv"
dffinbtc = pd.read_csv(file_path1)
dffineth = pd.read_csv(file_path2)
dffinsol = pd.read_csv(file_path3)
dffinxrp = pd.read_csv(file_path4)

print(dffinbtc.columns)

# =====================================
# ESTADÍSTICAS DESCRIPTIVAS
# =====================================

print("==== BITCOIN ====")
print(dffinbtc[["sentiment_score", "price"]].describe().round(4))

print("\n==== ETHEREUM ====")
print(dffineth[["sentiment_score", "price"]].describe().round(4))

print("\n==== SOLANA ====")
print(dffinsol[["sentiment_score", "price"]].describe().round(4))

print("\n==== XRP ====")
print(dffinxrp[["sentiment_score", "price"]].describe().round(4))

# para poder ver los estadísticos de manera visual en vez de en la consola

tabla_resumen = pd.concat([
    dffinbtc[["sentiment_score", "price"]].describe().round(4).add_suffix("_BTC"),
    dffineth[["sentiment_score", "price"]].describe().round(4).add_suffix("_ETH"),
    dffinsol[["sentiment_score", "price"]].describe().round(4).add_suffix("_SOL"),
    dffinxrp[["sentiment_score", "price"]].describe().round(4).add_suffix("_XRP")
], axis=1)

fig, ax = plt.subplots(figsize=(16, 4))
ax.axis("off")
ax.table(cellText=tabla_resumen.values, colLabels=tabla_resumen.columns,
         rowLabels=tabla_resumen.index, cellLoc="center", loc="center")
plt.show()

# =====================================
# GRAFICAMOS LA EVOLUCÓN DEL PRECIO
# =====================================


fig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0, 0].plot(dffinbtc["UTC_Time"], dffinbtc["price"])
axes[0, 0].set_title("Precio diario Bitcoin")
axes[0, 0].set_xlabel("Fecha")
axes[0, 0].set_ylabel("Precio")

axes[0, 1].plot(dffineth["UTC_Time"], dffineth["price"])
axes[0, 1].set_title("Ethereum Daily Price")
axes[0, 1].set_xlabel("Fecha")
axes[0, 1].set_ylabel("Precio")

axes[1, 0].plot(dffinsol["UTC_Time"], dffinsol["price"])
axes[1, 0].set_title("Solana Daily Price")
axes[1, 0].set_xlabel("Fecha")
axes[1, 0].set_ylabel("Precio")

axes[1, 1].plot(dffinxrp["UTC_Time"], dffinxrp["price"])
axes[1, 1].set_title("XRP Daily Price")
axes[1, 1].set_xlabel("Fecha")
axes[1, 1].set_ylabel("Precio")

plt.tight_layout()
plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL NÚMERO DE TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (BITCOIN)
# =====================================

# Agrupar por día
df_daily_tweets_price_BTC = (
    dffinbtc
    .groupby("UTC_Time")
    .agg(
        tweet_count=("Tweet_Content", "count"),
        price=("price", "first")
    )
    .reset_index()
)

print(df_daily_tweets_price_BTC.head())

fig, ax1 = plt.subplots(figsize=(12, 6))

# Puntos rojos para número de tweets
ax1.scatter(
    df_daily_tweets_price_BTC["UTC_Time"],
    df_daily_tweets_price_BTC["tweet_count"],
    color="red",
    label="Tweets por día"
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Número de tweets", color="red")

# Segundo eje para el precio
ax2 = ax1.twinx()

# Línea azul clara para el precio
ax2.plot(
    df_daily_tweets_price_BTC["UTC_Time"],
    df_daily_tweets_price_BTC["price"],
    color="skyblue",
    linewidth=2,
    label="Precio BTC"
)
ax2.set_ylabel("Precio de Bitcoin", color="skyblue")

plt.title("Número de tweets y precio diario de Bitcoin")
plt.xticks(rotation=45)
plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL NÚMERO DE TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (ETHEREUM)
# =====================================

# Agrupar por día
df_daily_tweets_price_ETH = (
    dffineth
    .groupby("UTC_Time")
    .agg(
        tweet_count=("Tweet_Content", "count"),
        price=("price", "first")
    )
    .reset_index()
)

print(df_daily_tweets_price_ETH.head())

fig, ax1 = plt.subplots(figsize=(12, 6))

# Puntos rojos para número de tweets
ax1.scatter(
    df_daily_tweets_price_ETH["UTC_Time"],
    df_daily_tweets_price_ETH["tweet_count"],
    color="red",
    label="Tweets por día"
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Número de tweets", color="red")

# Segundo eje para el precio
ax2 = ax1.twinx()

# Línea azul clara para el precio
ax2.plot(
    df_daily_tweets_price_ETH["UTC_Time"],
    df_daily_tweets_price_ETH["price"],
    color="skyblue",
    linewidth=2,
    label="Precio ETH"
)
ax2.set_ylabel("Precio de Ethereum", color="skyblue")

plt.title("Número de tweets y precio diario de Ethereum")
plt.xticks(rotation=45)
plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL NÚMERO DE TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (SOLANA)
# =====================================

# Agrupar por día
df_daily_tweets_price_SOL = (
    dffinsol
    .groupby("UTC_Time")
    .agg(
        tweet_count=("Tweet_Content", "count"),
        price=("price", "first")
    )
    .reset_index()
)

print(df_daily_tweets_price_SOL.head())

fig, ax1 = plt.subplots(figsize=(12, 6))

# Puntos rojos para número de tweets
ax1.scatter(
    df_daily_tweets_price_SOL["UTC_Time"],
    df_daily_tweets_price_SOL["tweet_count"],
    color="red",
    label="Tweets por día"
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Número de tweets", color="red")

# Segundo eje para el precio
ax2 = ax1.twinx()

# Línea azul clara para el precio
ax2.plot(
    df_daily_tweets_price_SOL["UTC_Time"],
    df_daily_tweets_price_SOL["price"],
    color="skyblue",
    linewidth=2,
    label="Precio SOL"
)
ax2.set_ylabel("Precio de Solana", color="skyblue")

plt.title("Número de tweets y precio diario de Solana")
plt.xticks(rotation=45)
plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL NÚMERO DE TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (XRP)
# =====================================

# Agrupar por día
df_daily_tweets_price_XRP = (
    dffinxrp
    .groupby("UTC_Time")
    .agg(
        tweet_count=("Tweet_Content", "count"),
        price=("price", "first")
    )
    .reset_index()
)

print(df_daily_tweets_price_XRP.head())

fig, ax1 = plt.subplots(figsize=(12, 6))

# Puntos rojos para número de tweets
ax1.scatter(
    df_daily_tweets_price_XRP["UTC_Time"],
    df_daily_tweets_price_XRP["tweet_count"],
    color="red",
    label="Tweets por día"
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Número de tweets", color="red")

# Segundo eje para el precio
ax2 = ax1.twinx()

# Línea azul clara para el precio
ax2.plot(
    df_daily_tweets_price_XRP["UTC_Time"],
    df_daily_tweets_price_XRP["price"],
    color="skyblue",
    linewidth=2,
    label="Precio XRP"
)
ax2.set_ylabel("Precio de XRP", color="skyblue")

plt.title("Número de tweets y precio diario de XRP")
plt.xticks(rotation=45)
plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL SENTIMIENTO DE LOS TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (BITCOIN)
# =====================================

fig, ax1 = plt.subplots(figsize=(12,6))

# Línea azul clara para el precio
ax1.plot(
    dffinbtc["UTC_Time"],
    dffinbtc["price"],
    color="skyblue",
    linewidth=2
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio de Bitcoin", color="skyblue")

# Segundo eje para el sentimiento
ax2 = ax1.twinx()

# Puntos verdes para el sentimiento
ax2.scatter(
    dffinbtc["UTC_Time"],
    dffinbtc["daily_sentiment_avg"],
    color="green",
    alpha=0.7
)
ax2.set_ylabel("Sentimiento medio diario", color="green")

plt.title("Precio de Bitcoin y sentimiento medio diario de tweets")
plt.xticks(rotation=45)

plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL SENTIMIENTO DE LOS TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (ETHEREUM)
# =====================================

fig, ax1 = plt.subplots(figsize=(12,6))

# Línea azul clara para el precio
ax1.plot(
    dffineth["UTC_Time"],
    dffineth["price"],
    color="skyblue",
    linewidth=2
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio de Ethereum", color="skyblue")

# Segundo eje para el sentimiento
ax2 = ax1.twinx()

# Puntos verdes para el sentimiento
ax2.scatter(
    dffineth["UTC_Time"],
    dffineth["daily_sentiment_avg"],
    color="green",
    alpha=0.7
)
ax2.set_ylabel("Sentimiento medio diario", color="green")

plt.title("Precio de Ethereum y sentimiento medio diario de tweets")
plt.xticks(rotation=45)

plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL SENTIMIENTO DE LOS TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (SOLANA)
# =====================================

fig, ax1 = plt.subplots(figsize=(12,6))

# Línea azul clara para el precio
ax1.plot(
    dffinsol["UTC_Time"],
    dffinsol["price"],
    color="skyblue",
    linewidth=2
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio de Solana", color="skyblue")

# Segundo eje para el sentimiento
ax2 = ax1.twinx()

# Puntos verdes para el sentimiento
ax2.scatter(
    dffinsol["UTC_Time"],
    dffinsol["daily_sentiment_avg"],
    color="green",
    alpha=0.7
)
ax2.set_ylabel("Sentimiento medio diario", color="green")

plt.title("Precio de Solana y sentimiento medio diario de tweets")
plt.xticks(rotation=45)

plt.show()

# =====================================
# GRAFICAMOS LA RELACIÓN DEL SENTIMIENTO DE LOS TWEETS POR DIA CON EL PRECIO DE LA CRYPTO (XRP)
# =====================================

fig, ax1 = plt.subplots(figsize=(12,6))

# Línea azul clara para el precio
ax1.plot(
    dffinxrp["UTC_Time"],
    dffinxrp["price"],
    color="skyblue",
    linewidth=2
)
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Precio de XRP", color="skyblue")

# Segundo eje para el sentimiento
ax2 = ax1.twinx()

# Puntos verdes para el sentimiento
ax2.scatter(
    dffinxrp["UTC_Time"],
    dffinxrp["daily_sentiment_avg"],
    color="green",
    alpha=0.7
)
ax2.set_ylabel("Sentimiento medio diario", color="green")

plt.title("Precio de XRP y sentimiento medio diario de tweets")
plt.xticks(rotation=45)

plt.show()


# =====================================
# DISTRIBUCIÓN DEL SENTIMIENTO POR CRYPTO
# =====================================

# Unimos los 4 df para comparar
dffinbtc["crypto"] = "BTC"
dffineth["crypto"] = "ETH"
dffinsol["crypto"] = "SOL"
dffinxrp["crypto"] = "XRP"

df_all = pd.concat([
    dffinbtc[["crypto", "sentiment_score"]],
    dffineth[["crypto", "sentiment_score"]],
    dffinsol[["crypto", "sentiment_score"]],
    dffinxrp[["crypto", "sentiment_score"]]
])

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_all, x="crypto", y="sentiment_score", palette="Set2")
plt.title("Distribución del sentimiento por criptomoneda")
plt.xlabel("Criptomoneda")
plt.ylabel("Sentiment Score")
plt.show()
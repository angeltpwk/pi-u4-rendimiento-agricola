# Proyecto Integrador U4 - Optimizacion Inferencial del Rendimiento Agricola
# Analisis y Exploracion de Datos (INC-2505) - Mariaca Palma Angel de Jesus - 23680062

import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Rendimiento Agrícola - PI U4", page_icon="🌱", layout="wide")

COLORES = {"Tratamiento_A": "#4C78A8", "Tratamiento_B": "#F58518"}
sns.set_theme(style="whitegrid")


# ---------- Hito 1: telemetria con DuckDB (semilla 0.42) ----------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("SELECT setseed(0.42)")
    con.execute("""
        CREATE TABLE telemetria_invernadero AS
        WITH base AS MATERIALIZED (
            SELECT
                i AS parcela_id,
                ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
                ROUND(50.0 + random() * 100.0, 2)  AS fertilizante_kg,
                ROUND(18.0 + random() * 14.0, 1)   AS temperatura_c,
                random()                           AS p_trat,
                (random() * 1.6 - 0.8)             AS ruido
            FROM generate_series(1, 100) AS t(i)
        )
        SELECT
            parcela_id, volumen_riego, fertilizante_kg, temperatura_c,
            CASE WHEN p_trat < 0.5 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            ROUND(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
                  + CASE WHEN p_trat < 0.5 THEN 0.0 ELSE 1.8 END + ruido, 2) AS rendimiento_ton
        FROM base
    """)
    return con.execute("SELECT * FROM telemetria_invernadero ORDER BY parcela_id").df()


df_total = cargar_datos()

# ---------- Barra lateral: filtros ----------
st.sidebar.header("Filtros")
trats = st.sidebar.multiselect(
    "Tipo de tratamiento",
    options=["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"],
)
rmin, rmax = float(df_total.volumen_riego.min()), float(df_total.volumen_riego.max())
rango = st.sidebar.slider("Volumen de riego (m³/ha)", rmin, rmax, (rmin, rmax))
st.sidebar.caption("Datos simulados con DuckDB, semilla 0.42, N = 100 parcelas.")

df = df_total[
    df_total.tipo_tratamiento.isin(trats)
    & df_total.volumen_riego.between(rango[0], rango[1])
].copy()

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.caption("Invernaderos inteligentes · Análisis y Exploración de Datos (INC-2505) · ITC / TecNM")

if len(df) < 10:
    st.warning("Con estos filtros quedan menos de 10 parcelas; amplía la selección para ajustar el modelo.")
    st.stop()

# ---------- Hito 2: regresion OLS (Riego + Fertilizante) ----------
X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
modelo = sm.OLS(df["rendimiento_ton"], X).fit()
df["residuo"] = modelo.resid
rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))

# ---------- Hito 3: t-Student, una cola (B > A) ----------
A = df.loc[df.tipo_tratamiento == "Tratamiento_A", "rendimiento_ton"]
B = df.loc[df.tipo_tratamiento == "Tratamiento_B", "rendimiento_ton"]
hay_prueba = len(A) >= 2 and len(B) >= 2
if hay_prueba:
    t_res = stats.ttest_ind(B, A, equal_var=True, alternative="greater")
    dif = B.mean() - A.mean()
    gl = len(A) + len(B) - 2
    sp2 = ((len(A) - 1) * A.var(ddof=1) + (len(B) - 1) * B.var(ddof=1)) / gl
    se = np.sqrt(sp2 * (1 / len(A) + 1 / len(B)))
    tcrit = stats.t.ppf(0.975, gl)
    ic = (dif - tcrit * se, dif + tcrit * se)

# ---------- KPIs ----------
k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento promedio", f"{df.rendimiento_ton.mean():.2f} ton/ha")
k2.metric("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.2f} %")
if hay_prueba:
    k3.metric("p-valor (t-Student, B > A)", f"{t_res.pvalue:.2e}",
              "Se rechaza H₀" if t_res.pvalue < 0.05 else "No se rechaza H₀", delta_color="off")
else:
    k3.metric("p-valor (t-Student, B > A)", "N/D", "Selecciona ambos tratamientos", delta_color="off")

st.divider()

# ---------- Hito 4: cuadricula 2x2 ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Mapa de calor de correlación (Pearson)")
    corr = df[["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]].corr()
    corr.index = corr.columns = ["Riego", "Fertilizante", "Temperatura", "Rendimiento"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
    st.pyplot(fig)
    r_riego, r_fert = corr.loc["Riego", "Rendimiento"], corr.loc["Fertilizante", "Rendimiento"]
    st.caption(f"Riego → Rendimiento: r = {r_riego:.2f}; Fertilizante → Rendimiento: r = {r_fert:.2f}. "
               f"El riego es el factor dominante.")

with c2:
    st.subheader("2. Dispersión con recta OLS (Riego vs Rendimiento)")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.scatterplot(data=df, x="volumen_riego", y="rendimiento_ton", hue="tipo_tratamiento",
                    palette=COLORES, ax=ax)
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", scatter=False, ci=95,
                line_kws={"color": "black"}, ax=ax)
    ax.set_xlabel("Volumen de riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    ax.legend(title="Tratamiento", fontsize=8)
    st.pyplot(fig)
    st.caption("Recta de mínimos cuadrados con banda de confianza al 95 %.")

c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Diagrama de cajas por tratamiento")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    orden = [t for t in ["Tratamiento_A", "Tratamiento_B"] if t in trats]
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                palette=COLORES, hue="tipo_tratamiento", legend=False, ax=ax)
    sns.pointplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                  errorbar=("ci", 95), color="black", linestyle="none", markers="D", ax=ax)
    ax.set_xlabel("Tipo de tratamiento")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)
    if hay_prueba:
        st.caption(f"Rombo negro = media con IC 95 %. Diferencia B − A = {dif:.2f} ton/ha, "
                   f"IC 95 % [{ic[0]:.2f}, {ic[1]:.2f}].")
    else:
        st.caption("Rombo negro = media con IC 95 %.")

with c4:
    st.subheader("4. Histograma y densidad de residuos")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.histplot(df["residuo"], bins=15, stat="density", kde=True, color="#4C78A8", ax=ax)
    xs = np.linspace(df.residuo.min() - 0.5, df.residuo.max() + 0.5, 200)
    ax.plot(xs, stats.norm.pdf(xs, 0, df.residuo.std()), "r--", label="Normal N(0, σ²)")
    ax.set_xlabel("Residuo = y real − y predicho (ton/ha)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption(f"RMSE = {rmse:.3f} ton/ha.")

st.divider()

# ---------- Verdicto ----------
st.subheader("Veredicto de ingeniería")
if hay_prueba:
    if t_res.pvalue < 0.05:
        st.success(f"p = {t_res.pvalue:.2e} < 0.05 → se rechaza H₀. El Tratamiento_B incrementa el rendimiento "
                   f"en promedio {dif:.2f} ton/ha (IC 95 %: {ic[0]:.2f} a {ic[1]:.2f}). "
                   f"Se recomienda escalar el bioestimulante.")
    else:
        st.info(f"p = {t_res.pvalue:.3f} ≥ 0.05 → no se rechaza H₀ con los datos filtrados.")
else:
    st.info("Selecciona ambos tratamientos para evaluar la prueba de hipótesis.")

with st.expander("Resumen estadístico del modelo OLS"):
    st.text(modelo.summary().as_text())
with st.expander("Ver datos (telemetría filtrada)"):
    st.dataframe(df.drop(columns="residuo"), width="stretch")

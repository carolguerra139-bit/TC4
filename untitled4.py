# ==========================================
# 🏥 APP CLÍNICO – APOIO À DECISÃO EM OBESIDADE
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier

# ------------------------------------------
# CONFIGURAÇÃO STREAMLIT
# ------------------------------------------
st.set_page_config(
    page_title="Apoio Clínico – Risco de Obesidade",
    layout="centered"
)

# ------------------------------------------
# TÍTULO E CONTEXTO CLÍNICO
# ------------------------------------------
st.title("🏥 Apoio à Decisão Clínica – Risco de Obesidade")

st.markdown("""
Este sistema utiliza **Inteligência Artificial** para **auxiliar médicos e médicas**
na avaliação do **risco de obesidade** em pacientes, com base em dados clínicos
e comportamentais.

⚠️ **Aviso importante**  
Este sistema **não substitui o julgamento clínico** nem o diagnóstico médico.
Seu uso é **exclusivamente como ferramenta de apoio à decisão**.
""")

st.divider()

# ------------------------------------------
# CARREGAMENTO DOS DADOS
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# ------------------------------------------
# VISÃO GERAL DO DATASET
# ------------------------------------------
with st.expander("📊 Visualizar amostra dos dados utilizados"):
    st.dataframe(df.head())

# ------------------------------------------
# DEFINIÇÃO DE FEATURES E TARGET
# ------------------------------------------
X = df.drop("Obesity", axis=1)
y = df["Obesity"]

num_features = ["Age", "Height", "Weight", "BMI"]
cat_features = [c for c in X.columns if c not in num_features]

# ------------------------------------------
# PRÉ-PROCESSAMENTO
# ------------------------------------------
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ("num", "passthrough", num_features)
])

# ------------------------------------------
# SPLIT TREINO / TESTE
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------
# MODELO – GRADIENT BOOSTING
# ------------------------------------------
model = Pipeline([
    ("prep", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])

model.fit(X_train, y_train)

# ------------------------------------------
# AVALIAÇÃO DO MODELO
# ------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📈 Desempenho do Modelo")

st.metric(
    "Precisão global do sistema",
    f"{acc:.1%}"
)

st.caption(
    f"➡️ Interpretação clínica: o sistema apresenta acerto médio em "
    f"{int(acc * 100)} a cada 100 pacientes avaliados."
)

st.divider()

# ------------------------------------------
# MATRIZ DE CONFUSÃO (INTERPRETAÇÃO CLÍNICA)
# ------------------------------------------
st.subheader("🔍 Comparação entre avaliação real e previsão do sistema")

st.markdown("""
Este gráfico mostra como o sistema se comporta em relação aos dados reais:

- **Acertos** indicam boa capacidade de triagem.
- **Erros** devem sempre ser analisados em conjunto com a avaliação clínica.
""")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax, cmap="Blues", values_format="d")
st.pyplot(fig)

st.divider()

# ------------------------------------------
# FATORES DE RISCO (FEATURE IMPORTANCE)
# ------------------------------------------
st.subheader("⚖️ Principais fatores associados ao risco de obesidade")

st.markdown("""
Os fatores abaixo foram identificados pelo modelo como os **mais relevantes**
na associação com obesidade na população analisada.

📌 Esses fatores **não representam causalidade direta**, mas **indicadores de risco**.
""")

gb_model = model.named_steps["model"]
feature_names = model.named_steps["prep"].get_feature_names_out()
importances = gb_model.feature_importances_

feat_imp = pd.DataFrame({
    "Fator": feature_names,
    "Importância": importances
}).sort_values(by="Importância", ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.barh(feat_imp["Fator"], feat_imp["Importância"])
ax2.invert_yaxis()
ax2.set_xlabel("Importância relativa")
ax2.set_title("Fatores mais associados à obesidade")
st.pyplot(fig2)

st.divider()

# ------------------------------------------
# AVALIAÇÃO INDIVIDUAL DO PACIENTE
# ------------------------------------------
st.subheader("🧍 Avaliação individual de paciente")

st.markdown("""
Preencha os dados abaixo para obter uma **estimativa de risco**.
""")

with st.form("form_paciente"):
    idade = st.number_input("Idade (anos)", 5, 100, 30)
    altura = st.number_input("Altura (m)", 1.20, 2.20, 1.70)
    peso = st.number_input("Peso (kg)", 20.0, 250.0, 70.0)

    dados_cat = {}
    for col in cat_features:
        dados_cat[col] = st.selectbox(col, df[col].unique())

    submitted = st.form_submit_button("Avaliar risco")

if submitted:
    bmi = peso / (altura ** 2)

    paciente = {
        "Age": idade,
        "Height": altura,
        "Weight": peso,
        "BMI": bmi,
        **dados_cat
    }

    paciente_df = pd.DataFrame([paciente])

    pred = model.predict(paciente_df)[0]

    st.subheader("🩺 Resultado da avaliação")

    st.write(f"**Classificação estimada:** {pred}")

    st.caption(
        "➡️ Esta estimativa deve ser interpretada em conjunto com "
        "avaliação clínica, exames e histórico do paciente."
    )

st.divider()

# ------------------------------------------
# RELATÓRIO TÉCNICO (OPCIONAL)
# ------------------------------------------
with st.expander("📄 Relatório técnico detalhado (opcional)"):
    st.text(classification_report(y_test, y_pred))

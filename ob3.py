import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try to import LightGBM but don't fail if not installed
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# Single page config call
st.set_page_config(layout="wide", page_title="Obesity Classification App")

# --- Helper functions (diagnostics and safe split) ---
def diagnosticos(X, y, top=20):
    """Gera um dicionário com diagnósticos sem modificar X ou y e escreve no Streamlit."""
    y_ser = pd.Series(y)
    info = {
        "X_shape": getattr(X, "shape", None),
        "y_shape": y_ser.shape,
        "y_nunique": int(y_ser.nunique()),
        "y_value_counts_head": y_ser.value_counts().head(top),
        "y_nans": int(y_ser.isna().sum())
    }
    st.write("### Diagnóstico")
    st.write("X.shape:", info["X_shape"])
    st.write("y.shape:", info["y_shape"])
    st.write("y unique:", info["y_nunique"])
    st.write("NaNs em y:", info["y_nans"])
    st.write(f"Contagens (top {top}):")
    st.write(info["y_value_counts_head"])
    return info

def make_stratify_bins(y, max_bins=10):
    """
    Tenta criar bins por quantis (pd.qcut) garantindo pelo menos 2 amostras por bin.
    Retorna um objeto categórico alinhado a y ou None se não for possível.
    """
    y_ser = pd.Series(y).reset_index(drop=True)
    n_bins = min(max_bins, y_ser.nunique())
    while n_bins > 1:
        try:
            bins = pd.qcut(y_ser, q=n_bins, duplicates='drop')
            vc = bins.value_counts()
            if (vc >= 2).all():
                return bins
        except Exception:
            pass
        n_bins -= 1
    return None

def safe_train_test_split(X, y, test_size=0.2, random_state=42, stratify_if_possible=True):
    """
    Tenta:
     1) split com stratify=y (se aplicável)
     2) se falhar, cria bins para estratificar (quando y é contínuo)
     3) fallback: split sem estratify
    Não modifica X ou y originais.
    Retorna X_train, X_test, y_train, y_test
    """
    y_ser = pd.Series(y).reset_index(drop=True)
    strat = None
    if stratify_if_possible and y_ser.nunique() > 1:
        strat = y_ser

    try:
        return train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=strat)
    except ValueError as e:
        st.warning("train_test_split falhou com stratify=%s : %s" % ("y" if strat is not None else "None", e))

    y_bins = make_stratify_bins(y_ser, max_bins=10)
    if y_bins is not None:
        try:
            st.info(f"Usando bins (n_bins={y_bins.nunique()}) para estratificar.")
            return train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y_bins)
        except Exception as e:
            st.warning("Split com bins também falhou: %s" % e)

    st.info("Fallback: realizando split sem estratificação.")
    return train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=None)


# ---------- Streamlit UI ----------
st.title("Obesity Classification — Streamlit")
st.markdown("App convertido do notebook obesidade_v5.ipynb — pré-processamento, treino e previsões interativas.")

# Sidebar: file upload and parameters
st.sidebar.header("Dados e Parâmetros")
uploaded_file = st.sidebar.file_uploader("Enviar CSV com os dados (ex: Obesity.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV enviado: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("Obesity.csv")
    except FileNotFoundError:
        st.error("Arquivo 'Obesity.csv' não encontrado e nenhum arquivo foi enviado. Faça upload do CSV.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler 'Obesity.csv': {e}")
        st.stop()

st.subheader("Visualização dos dados")
st.write("Dimensões:", df.shape)
st.dataframe(df.head())

# Ensure IMC column exists (original notebook computed IMC = Weight / Height**2)
if "IMC" not in df.columns and {"Weight", "Height"}.issubset(df.columns):
    # Assume Height is in meters — mirror do notebook
    with st.spinner("Calculando IMC (Weight / Height**2)..."):
        # avoid division by zero
        height_safe = df["Height"].replace({0: np.nan})
        df["IMC"] = df["Weight"] / (height_safe ** 2)
        st.success("Coluna IMC adicionada.")

# Select target column (by default last column)
target_default_index = len(df.columns) - 1
target_col = st.sidebar.selectbox("Selecione a coluna alvo (classe)", options=df.columns.tolist(), index=target_default_index)

# Show class distribution
st.subheader("Distribuição das classes")
st.write(df[target_col].value_counts())
fig, ax = plt.subplots()
df[target_col].value_counts().plot(kind="bar", ax=ax)
ax.set_ylabel("Count")
st.pyplot(fig)

# Identify categorical and numeric features (exclude target)
all_features = df.columns.drop(target_col).tolist()
# Treat object and category dtypes as categorical
categorical_cols = df[all_features].select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [c for c in all_features if c not in categorical_cols]

st.sidebar.write(f"Colunas detectadas: {len(all_features)} (numéricas: {len(numeric_cols)}, categóricas: {len(categorical_cols)})")

st.subheader("Colunas utilizadas")
st.write("Numéricas:", numeric_cols)
st.write("Categóricas:", categorical_cols)

# Train/test split parameter
test_size = st.sidebar.slider("Proporção de teste", min_value=0.05, max_value=0.5, value=0.25, step=0.05)
random_state = int(st.sidebar.number_input("random_state", value=0, step=1))

# Model selection
st.sidebar.header("Modelo e hiperparâmetros")
model_options = ["Decision Tree", "Random Forest"]
if HAS_LGB:
    model_options.append("LightGBM")
model_name = st.sidebar.selectbox("Escolha o modelo", options=model_options)

if model_name == "Decision Tree":
    max_depth = st.sidebar.number_input("max_depth (Decision Tree, 0 = None)", min_value=0, value=0, step=1)
    criterion = st.sidebar.selectbox("criterion", options=["gini", "entropy"], index=1)
    def build_model():
        md = None if int(max_depth) == 0 else int(max_depth)
        return DecisionTreeClassifier(criterion=criterion, max_depth=md, random_state=random_state)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.number_input("n_estimators (Random Forest)", min_value=1, value=100, step=1)
    max_depth = st.sidebar.number_input("max_depth (Random Forest, 0 = None)", min_value=0, value=0, step=1)
    def build_model():
        md = None if int(max_depth) == 0 else int(max_depth)
        return RandomForestClassifier(n_estimators=int(n_estimators), max_depth=md, random_state=random_state)
else:  # LightGBM
    n_estimators = st.sidebar.number_input("n_estimators (LightGBM)", min_value=1, value=100, step=1)
    learning_rate = st.sidebar.number_input("learning_rate (LightGBM)", min_value=0.001, value=0.1, format="%.3f")
    def build_model():
        return lgb.LGBMClassifier(n_estimators=int(n_estimators), learning_rate=float(learning_rate), random_state=random_state)

# Preprocessor: OneHot for categoricals, StandardScaler for numerics
# Use sparse_output if available, otherwise sparse for compatibility
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_cols) if len(categorical_cols) > 0 else (),
        ("num", scaler, numeric_cols) if len(numeric_cols) > 0 else (),
    ],
    remainder='drop'
)

# Prepare X, y
X = df[all_features].copy()
y = df[target_col].copy()

# Optional diagnostics checkbox
if st.checkbox("Ativar diagnóstico (imprime value_counts, NaNs, etc.)", value=False):
    diagnosticos(X, y)

# Split (using safe function)
try:
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_size, random_state=random_state, stratify_if_possible=True)
except Exception as e:
    st.error(f"Erro ao realizar train_test_split: {e}")
    st.stop()

# Build pipeline
clf = build_model()
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

# Train button
if st.button("Treinar modelo"):
    with st.spinner("Treinando..."):
        try:
            pipeline.fit(X_train, y_train)
            # persist fitted pipeline in session_state
            st.session_state['pipeline'] = pipeline
            st.session_state['is_trained'] = True
            st.success("Treinamento concluído.")
        except Exception as e:
            st.error(f"Erro durante o treinamento: {e}")

# If a trained pipeline is available in session_state, use it for metrics and prediction
if st.session_state.get('is_trained', False) and 'pipeline' in st.session_state:
    pipeline_fitted = st.session_state['pipeline']

    # Predictions on test
    try:
        y_pred = pipeline_fitted.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Métricas no conjunto de teste")
        st.write(f"Accuracy: {acc:.4f}")
        st.text("Relatório de classificação:")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        labels = np.unique(y)  # ensure consistent ordering
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig_cm)
    except Exception as e:
        st.warning(f"Não foi possível calcular métricas/predição: {e}")

    # Feature names extraction
    try:
        # scikit-learn >=1.0 has get_feature_names_out on ColumnTransformer
        feature_names = pipeline_fitted.named_steps['preprocessor'].get_feature_names_out()
        # Clean names produced by ColumnTransformer if needed
        feature_names = [str(fn).replace("cat__", "").replace("num__", "") for fn in feature_names]
    except Exception:
        # Fallback: construct from categories_ and numeric cols
        feature_names = []
        try:
            cat_transformer = pipeline_fitted.named_steps['preprocessor'].named_transformers_.get('cat', None)
            if cat_transformer is not None and hasattr(cat_transformer, 'categories_'):
                cats = cat_transformer.categories_
                cat_cols = categorical_cols
                for col, categories in zip(cat_cols, cats):
                    feature_names.extend([f"{col}_{c}" for c in categories])
        except Exception:
            pass
        feature_names.extend(numeric_cols)

    st.write("Número de features após pré-processamento:", len(feature_names))

    # Feature importances for tree-based models
    model_after = pipeline_fitted.named_steps['clf']
    if hasattr(model_after, "feature_importances_"):
        importances = model_after.feature_importances_
        # Align to feature names (if lengths match)
        if len(importances) == len(feature_names):
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(20)
            st.subheader("Top features (feature_importances_)")
            st.dataframe(imp_df)
            fig_imp, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="importance", y="feature", data=imp_df, ax=ax)
            st.pyplot(fig_imp)
        else:
            st.warning("Não foi possível alinhar feature_importances_ com feature names (length mismatch).")
    else:
        st.info("O modelo escolhido não expõe feature_importances_.")

    # If Decision Tree, plot the tree
    if isinstance(model_after, DecisionTreeClassifier):
        st.subheader("Árvore de decisão (visualização)")
        fig_tree, ax = plt.subplots(figsize=(20, 12))
        try:
            plot_tree(model_after, feature_names=feature_names, class_names=[str(c) for c in np.unique(y)], filled=True, rounded=True, fontsize=8, ax=ax)
            st.pyplot(fig_tree)
        except Exception as e:
            st.warning(f"Falha ao desenhar árvore: {e}")
else:
    st.info("Treine o modelo (clique em 'Treinar modelo') para ver métricas e importâncias.")

# Single-sample prediction form
st.sidebar.header("Previsão — amostra única")
with st.sidebar.form("predict_form"):
    st.write("Preencha as features para prever a classe")
    sample_inputs = {}
    for col in all_features:
        if col in numeric_cols:
            col_min = float(df[col].min()) if np.isfinite(df[col].min()) else -1e6
            col_max = float(df[col].max()) if np.isfinite(df[col].max()) else 1e6
            col_mean = float(df[col].mean()) if np.isfinite(df[col].mean()) else 0.0
            step = (col_max - col_min) / 100.0 if col_max > col_min else 1.0
            sample_inputs[col] = st.number_input(col, value=col_mean, min_value=col_min, max_value=col_max, step=step)
        else:
            # categorical
            options = df[col].dropna().unique().tolist()
            if len(options) == 0:
                options = [""]
            sample_inputs[col] = st.selectbox(col, options=options)
    submit_predict = st.form_submit_button("Prever")

if submit_predict:
    if not st.session_state.get('is_trained', False) or 'pipeline' not in st.session_state:
        st.error("O pipeline não foi treinado ainda. Clique em 'Treinar modelo' antes de prever.")
    else:
        pipeline_fitted = st.session_state['pipeline']
        sample_df = pd.DataFrame([sample_inputs])
        try:
            pred = pipeline_fitted.predict(sample_df)[0]
            st.sidebar.success(f"Predição: {pred}")
            # Probabilities if available
            if hasattr(pipeline_fitted.named_steps['clf'], "predict_proba"):
                prob = pipeline_fitted.predict_proba(sample_df)[0]
                classes = pipeline_fitted.named_steps['clf'].classes_
                prob_df = pd.DataFrame({"class": classes, "probability": prob})
                st.sidebar.dataframe(prob_df.sort_values("probability", ascending=False).reset_index(drop=True))
        except Exception as e:
            st.sidebar.error(f"Erro ao prever: {e}")

st.markdown("---")
st.markdown("Observações:")
st.markdown("""
- O pré-processamento usa OneHotEncoder (handle_unknown='ignore') e StandardScaler.
- O pipeline treina o classificador escolhido diretamente; a extração de nomes de features usa get_feature_names_out quando disponível.
- Se quiser persistir o modelo, basta exportar pipeline com pickle em um bloco adicional.
""")
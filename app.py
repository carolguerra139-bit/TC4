import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from typing import Optional

st.set_page_config(page_title="Obesidade — Predição + Explicação (SHAP)", layout="wide")

# TRADUÇÕES 
TRADUCAO_TARGET = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I',
    'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III'
}

LABELS_FAF = {0: "Sedentário", 1: "1-2x/sem", 2: "3-4x/sem", 3: "4+x/sem"}
LABELS_NCP = {1: "1 Ref/dia", 2: "2 Ref/dia", 3: "3 Ref/dia", 4: "4+ Ref/dia"}
GENERO_OPT = {"Female": "Feminino", "Male": "Masculino"}

FRIENDLY_NAMES_SHAP = {
    "IMC": "IMC", "Idade": "Idade", "freq_semanal_atividade_fisica": "Ativ. Física",
    "num_refeicoes_princ": "Qtd. Refeições", "Genero": "Gênero"
}

# Dicionários para Aba 2
LABELS_TRANS = {"Automobile": "Carro", "Motorbike": "Moto", "Bike": "Bicicleta", "Public_Transportation": "Transp. Público", "Walking": "A pé"}
LABELS_FREQ = {"no": "Não", "Sometimes": "Às vezes", "Frequently": "Frequentemente", "Always": "Sempre"}
BINARY_OPT = {"yes": "Sim", "no": "Não"}

# FUNÇÕES 
@st.cache_data
def carregar_dados(path: str = "Obesity.csv") -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    
    target_col = None
    if 'Obesity' in df.columns: target_col = 'Obesity'
    elif 'NObeyesdad' in df.columns: target_col = 'NObeyesdad'
    elif 'Nivel_Obesidade' in df.columns: target_col = 'Nivel_Obesidade'
        
    if target_col:
        df['Obesity_pt'] = df[target_col].map(TRADUCAO_TARGET).fillna(df[target_col])
        df['Diagnóstico'] = df['Obesity_pt']
    else:
        df['Diagnóstico'] = 'Desconhecido'
    
    mapa_renomeacao = {
        'Age': 'Age', 'Gender': 'Gender',
        'FCVC': 'fc_vegetais', 'NCP': 'n_refeicoes',
        'CH2O': 'agua_diaria', 'family_history_with_overweight': 'hist_familiar',
        'FAF': 'ativ_fisica', 'TUE': 'tempo_telas', 'CALC': 'alcool',
        'SMOKE': 'fuma', 'CAEC': 'comer_entre_ref', 'MTRANS': 'transporte'
    }
    mapa_valido = {k: v for k, v in mapa_renomeacao.items() if k in df.columns}
    df = df.rename(columns=mapa_valido)
    df['Gênero'] = df['Gender'].map(GENERO_OPT).fillna(df.get('Gender'))
    df['IMC_Calc'] = df['Weight'] / (df['Height'] ** 2)
    return df

@st.cache_resource
def carregar_modelo(path: str = "modelo_obesidade.pkl"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

def montar_entrada(imc, idade, genero_raw, faf, ncp) -> pd.DataFrame:
    return pd.DataFrame({
        'IMC': [float(imc)],
        'Idade': [int(idade)],
        'Genero': [genero_raw],
        'freq_semanal_atividade_fisica': [int(faf)],
        'num_refeicoes_princ': [int(ncp)]
    })

def clean_name_shap(name):
    n = name.split('__')[-1] if '__' in name else name
    if "Genero" in n or "Gênero" in n: return "Gênero"
    return FRIENDLY_NAMES_SHAP.get(n, n)

def agrupar_shap(feat_names, shap_values):
    agrupado = {}
    for name, val in zip(feat_names, shap_values):
        clean = clean_name_shap(name)
        agrupado.setdefault(clean, 0.0)
        agrupado[clean] += val
    return pd.DataFrame(list(agrupado.items()), columns=['feature', 'shap_value'])

# APP 
df = carregar_dados()
modelo = carregar_modelo()

preprocessor = None
classifier = None
if modelo and hasattr(modelo, "named_steps"):
    preprocessor = modelo.named_steps.get('preprocessor')
    classifier = modelo.named_steps.get('classifier')

explainer = None
if classifier:
    try:
        explainer = shap.TreeExplainer(classifier)
    except:
        explainer = None

st.title("🏥 Sistema de Diagnóstico (IA Explicável)")
st.markdown("---")

tab1, tab2 = st.tabs(["🩺 Análise Individual", "📊 Visão Estratégica & Curiosidades"])


# ABA 1: 

with tab1:
    if not modelo:
        st.error("Modelo 'modelo_obesidade.pkl' não encontrado na pasta.")
    else:
        c_in, c_out = st.columns([1, 1.3])
        with c_in:
            st.subheader("Paciente")
            with st.form("main_form"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    idade = st.number_input("Idade", 2, 100, 36)
                    peso = st.number_input("Peso (kg)", 5.0, 200.0, 69.0)
                with cc2:
                    altura = st.number_input("Altura (m)", 0.50, 2.50, 1.64)
                    genero = st.selectbox("Gênero", ["Female", "Male"], format_func=lambda x: GENERO_OPT[x])
                
                st.markdown("##### Estilo de Vida")
                faf = st.selectbox("Ativ. Física", list(LABELS_FAF.keys()), format_func=lambda x: LABELS_FAF[x])
                ncp = st.selectbox("Refeições/dia", list(LABELS_NCP.keys()), format_func=lambda x: LABELS_NCP[x])
                
                btn_calc = st.form_submit_button("Calcular Risco", type="primary")

        with c_out:
            if btn_calc:
                imc = peso / (altura ** 2)
                X_in = montar_entrada(imc, idade, genero, faf, ncp)
                
                try:
                    # 1. Predição
                    raw = modelo.predict(X_in)[0]
                    prob_arr = modelo.predict_proba(X_in)
                    idx = list(classifier.classes_).index(raw)
                    prob = prob_arr[0][idx]
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    st.stop()
                
                res_pt = TRADUCAO_TARGET.get(raw, raw)
                
                # 2. Definição de Cores do Card
                color = "#17a2b8"
                if 'Obesity' in str(raw): color = "#dc3545"      # Vermelho
                elif 'Overweight' in str(raw): color = "#ffc107" # Amarelo
                elif 'Normal' in str(raw): color = "#28a745"     # Verde
                elif 'Insufficient' in str(raw): color = "#6c757d" # Cinza

                # 3. Lógica do Semáforo 
                # Define texto e ícone baseado na certeza, sem mostrar número grande
                if prob > 0.75:
                    txt_confianca = "Alta Confiança"
                    icone_conf = "🔒"
                elif prob > 0.50:
                    txt_confianca = "Confiança Moderada"
                    icone_conf = "⚖️"
                else:
                    txt_confianca = "Caso Limítrofe (Atenção)"
                    icone_conf = "⚠️"

                # 4. Renderização HTML Limpa
                # O 'span title' cria o tooltip: o número só aparece se passar o mouse
                st.markdown(f"""
                <div style="background-color:{color}25; padding:20px; border-radius:10px; border-left:8px solid {color}; margin-bottom:20px">
                    <h3 style="color:{color}; margin:0">Resultado: {res_pt}</h3>
                    <p style="margin:8px 0 0 0; font-size:1.1em">
                        IMC: <b>{imc:.2f}</b> • 
                        <span title="Certeza estatística exata: {prob*100:.1f}%" style="cursor:help; border-bottom:1px dotted #666;">
                            {icone_conf} {txt_confianca}
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # LÓGICA DO SHAP 
                if explainer and preprocessor:
                    try:
                        X_trans = preprocessor.transform(X_in)
                        if not isinstance(X_trans, np.ndarray): X_trans = X_trans.toarray()
                        try: raw_names = list(preprocessor.get_feature_names_out())
                        except: raw_names = [f"Var {i}" for i in range(X_trans.shape[1])]
                        
                        shap_vals = explainer.shap_values(X_trans)
                        
                        if isinstance(shap_vals, list): sv = shap_vals[idx][0]
                        else:
                            if len(shap_vals.shape) == 3:
                                try: sv = shap_vals[0, :, idx]
                                except: sv = shap_vals[idx][0]
                            elif len(shap_vals.shape) == 2: sv = shap_vals[0]
                            else: sv = shap_vals
                        
                        sv = np.array(sv).flatten()
                        if len(raw_names) != len(sv):
                            raw_names = raw_names[:len(sv)] if len(raw_names) > len(sv) else raw_names + [f"X{i}" for i in range(len(sv)-len(raw_names))]
                        
                        df_shap = agrupar_shap(raw_names, sv)
                        df_shap["Impacto"] = df_shap["shap_value"].apply(lambda x: "Aumenta Chance" if x > 0 else "Diminui Chance")
                        df_shap["abs_val"] = df_shap["shap_value"].abs()
                        df_shap = df_shap.sort_values(by="abs_val", ascending=True)
                        
                        # Cores inteligentes: Verde=Bom, Vermelho=Ruim
                        is_healthy = (res_pt == 'Peso Normal')
                        if is_healthy:
                            mapa_cores = {"Aumenta Chance": "#28a745", "Diminui Chance": "#dc3545"}
                        else:
                            mapa_cores = {"Aumenta Chance": "#dc3545", "Diminui Chance": "#28a745"}

                        st.subheader("🔍 O que pesou na decisão?")
                        fig = px.bar(
                            df_shap, 
                            x="shap_value", 
                            y="feature", 
                            orientation='h', 
                            color="Impacto",
                            color_discrete_map=mapa_cores, 
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("**Interpretação Completa:**")
                        df_txt = df_shap.sort_values(by="shap_value", ascending=False)
                        
                        for _, row in df_txt.iterrows():
                            val = row['shap_value']
                            nome = row['feature']
                            
                            valor_usuario = ""
                            if nome == "IMC": valor_usuario = f"({imc:.1f})"
                            elif nome == "Idade": valor_usuario = f"({idade})"
                            elif nome == "Gênero": valor_usuario = f"({GENERO_OPT[genero]})"
                            
                            if val > 0.01: 
                                if is_healthy:
                                    st.success(f"✅ **{nome} {valor_usuario}**: Contribuiu para o diagnóstico de Peso Normal.")
                                else:
                                    st.warning(f"🔺 **{nome} {valor_usuario}**: Aumentou a chance deste diagnóstico.")
                            elif val < -0.01: 
                                if is_healthy:
                                    st.warning(f"⚠️ **{nome} {valor_usuario}**: Diminuiu a certeza de Peso Normal (Fator de Risco).")
                                else:
                                    st.success(f"🛡️ **{nome} {valor_usuario}**: Protegeu (diminuiu a chance do problema).")
                            else: 
                                st.info(f"⚪ **{nome}**: Pouco impacto na decisão.")
                                
                    except Exception as ex:
                        st.error(f"Erro ao gerar explicação SHAP: {ex}")


# ABA 2: Visão Estratégica & Curiosidades

with tab2:
    if df is not None:
        MAPA_VISUALIZACAO = {
            "IMC": {"col": "IMC_Calc", "tipo": "num", "unit": "kg/m²", "title": "IMC Médio"},
            "Idade": {"col": "Age", "tipo": "num", "unit": "anos", "title": "Idade Média"},
            "Qtd. Refeições": {"col": "n_refeicoes", "tipo": "num", "unit": "ref/dia", "title": "Média de Refeições"},
            "Vegetais": {"col": "fc_vegetais", "tipo": "num", "unit": "escala 1-3", "title": "Consumo de Vegetais"},
            "Água": {"col": "agua_diaria", "tipo": "num", "unit": "litros", "title": "Consumo de Água"},
            "Telas": {"col": "tempo_telas", "tipo": "num", "unit": "horas", "title": "Tempo de Tela"},
            "Gênero": {"col": "Gender", "tipo": "cat", "title": "Distribuição por Gênero"},
            "Ativ. Física": {"col": "ativ_fisica", "tipo": "cat", "title": "Frequência de Exercícios"},
            "Histórico Fam.": {"col": "hist_familiar", "tipo": "cat", "title": "Histórico Familiar"},
            "Transporte": {"col": "transporte", "tipo": "cat", "title": "Meio de Transporte"},
            "Fumar": {"col": "fuma", "tipo": "cat", "title": "Tabagismo"},
            "Álcool": {"col": "alcool", "tipo": "cat", "title": "Consumo de Álcool"},
            "Beliscar": {"col": "comer_entre_ref","tipo": "cat", "title": "Hábito de Beliscar"}
        }

        st.markdown("### 🧬 Análise Focada: Top 5 Fatores")
        st.info("Os gráficos abaixo detalham **apenas** as 5 variáveis que a IA identificou como cruciais acima.")
        
        diagnosticos = st.multiselect(
            "Grupos para comparar:",
            df['Diagnóstico'].unique(),
            default=['Peso Normal', 'Obesidade I', 'Obesidade III']
        )
        st.markdown("---")

        if diagnosticos:
            dff = df[df['Diagnóstico'].isin(diagnosticos)].copy()
            if 'IMC_Calc' not in dff.columns and 'Weight' in dff.columns:
                 dff['IMC_Calc'] = dff['Weight'] / (dff['Height']**2)

            ordem_logica = ['Abaixo do Peso', 'Peso Normal', 'Sobrepeso I', 'Sobrepeso II', 'Obesidade I', 'Obesidade II', 'Obesidade III']
            cat_orders = {"Diagnóstico": [x for x in ordem_logica if x in diagnosticos]}

            top_5_names = []
            if classifier and hasattr(classifier, 'feature_importances_'):
                try:
                    if preprocessor: feat_names_raw = preprocessor.get_feature_names_out()
                    else: feat_names_raw = [f"Var {i}" for i in range(len(classifier.feature_importances_))]

                    df_imp = pd.DataFrame({'Feature': feat_names_raw, 'Importance': classifier.feature_importances_})
                    df_imp['Nome Amigavel'] = df_imp['Feature'].apply(lambda x: clean_name_shap(x))
                    
                    df_imp_grouped = df_imp.groupby('Nome Amigavel')['Importance'].sum().reset_index()
                    df_imp_grouped = df_imp_grouped.sort_values(by='Importance', ascending=True).tail(5)
                    
                    top_5_names = df_imp_grouped['Nome Amigavel'].tolist()[::-1]

                    st.markdown("#### ⭐ O que define o peso nesses casos?")
                    fig_imp = px.bar(
                        df_imp_grouped, x='Importance', y='Nome Amigavel', orientation='h',
                        text_auto='.1%', title="", color_discrete_sequence=['#5A67D8']
                    )
                    fig_imp.update_layout(yaxis_title="", xaxis_title="Peso na Decisão", height=250)
                    fig_imp.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.markdown("---")
                except Exception as e: st.error(f"Erro Top 5: {e}")
            
            if not top_5_names: top_5_names = ["IMC", "Idade", "Gênero"]

            st.subheader("🔎 Detalhe das Variáveis Principais")
            col1, col2 = st.columns(2)
            

            # Esta paleta é baseada nas cores Pastel do Plotly, mas fixa para que não haja confusão na legenda.
            COLOR_PALETTE = px.colors.qualitative.Pastel
            
            for i, nome_variavel in enumerate(top_5_names):
                config = MAPA_VISUALIZACAO.get(nome_variavel)
                if config:
                    col_code = config['col']
                    tipo = config['tipo']
                    titulo = config['title']
                    onde_exibir = col1 if i % 2 == 0 else col2
                    
                    with onde_exibir:
                        if tipo == 'num':
                            # GRÁFICOS DE MÉDIAS (IMC, Idade, Refeições)
                            if col_code in dff.columns:
                                df_mean = dff.groupby('Diagnóstico')[col_code].mean().reset_index()
                                
                      
                                # Isso garante que as cores das barras sigam a ordem lógica e não causem confusão.
                                color_map = {diag: COLOR_PALETTE[cat_orders['Diagnóstico'].index(diag) % len(COLOR_PALETTE)] 
                                             for diag in cat_orders['Diagnóstico']}
                                df_mean['color'] = df_mean['Diagnóstico'].map(color_map)
                                
                                fig = px.bar(
                                    df_mean, 
                                    x='Diagnóstico', 
                                    y=col_code,
                                         category_orders=cat_orders,
                                    text_auto='.1f'
                                )
                                # Define as cores das barras manualmente
                                fig.update_traces(marker_color=df_mean['color'])


                                fig.update_layout(
                                    title=dict(text=f"<b>{titulo}</b>", font=dict(size=14)),
                                    yaxis_title=config.get('unit', ''), 
                                    xaxis_title="", 
                                    showlegend=False, 
                                    height=280,
                                    margin=dict(l=20, r=20, t=60, b=20)
                                )
                                fig.update_xaxes(showticklabels=True)
                                st.plotly_chart(fig, use_container_width=True)

                        elif tipo == 'cat':

                            if col_code in dff.columns:
                                dff_plot = dff.copy()
                                if col_code == 'ativ_fisica': dff_plot[col_code] = dff_plot[col_code].round().map(LABELS_FAF)
                                elif col_code == 'transporte': dff_plot[col_code] = dff_plot[col_code].str.strip().map(LABELS_TRANS).fillna(dff_plot[col_code])
                                elif col_code in ['fuma', 'hist_familiar']: dff_plot[col_code] = dff_plot[col_code].str.strip().map(BINARY_OPT).fillna(dff_plot[col_code])
                                elif col_code in ['alcool', 'comer_entre_ref']: dff_plot[col_code] = dff_plot[col_code].str.strip().map(LABELS_FREQ).fillna(dff_plot[col_code])
                                elif col_code == 'Gender': dff_plot['Gender'] = dff_plot['Gender'].map(GENERO_OPT)

                                fig = px.histogram(
                                    dff_plot, y="Diagnóstico", color=col_code,
                                    barnorm='percent', text_auto='.0f',
                                    category_orders=cat_orders, orientation='h',
                                    color_discrete_sequence=px.colors.qualitative.Safe
                                )

                                fig.update_layout(
                                    title=dict(text=f"<b>{titulo}</b>", font=dict(size=14)),
                                    yaxis_title="", xaxis_title="%",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    showlegend=True, 
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
                                    height=280, 
                                    margin=dict(l=0, r=0, t=60, b=0)
                                )
                                fig.update_traces(texttemplate='%{value:.0f}%', textposition='inside')
                                fig.update_xaxes(showticklabels=False)
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selecione pelo menos um grupo acima.")
    else:
        st.error("Dados não carregados.")
        #cd "Desktop\Data Analytics\Tech Challenge\4ª Fase\Modelo" streamlit run app.py
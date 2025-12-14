import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import warnings
import joblib  # salvar o modelo

# não mostrar mensagens de avisos
warnings.filterwarnings('ignore')

#  1. Carregar, Renomear e Preparar os Dados 
try:
    df = pd.read_csv("Obesity.csv")
except FileNotFoundError:
    print("Erro: Arquivo 'Obesity.csv' não encontrado.")
    exit()

# Dicionário
mapa_colunas = {
    'Gender': 'Genero',
    'Age': 'Idade',
    'Height': 'Altura',
    'Weight': 'Peso',
    'family_history': 'historico_familiar',
    'FAVC': 'cons_caloricos_freq',
    'FCVC': 'cons_vegetais_freq',
    'NCP': 'num_refeicoes_princ',
    'CAEC': 'cons_lanches_entre_refeicoes',
    'SMOKE': 'fuma',
    'CH2O': 'cons_agua_diaria',
    'SCC': 'monitora_calorias_diarias',
    'FAF': 'freq_semanal_atividade_fisica',
    'TUE': 'tempo_tecnologia_diario',
    'CALC': 'bebe_alcool',
    'MTRANS': 'meio_transporte_habitual',
    'Obesity': 'Nivel_Obesidade' # coluna alvo
}
# RENOMEAR AS COLUNAS
df.rename(columns=mapa_colunas, inplace=True)

# Arredondar conforme dicionário
colunas_para_arredondar = [
    'cons_vegetais_freq', 
    'num_refeicoes_princ', 
    'cons_agua_diaria', 
    'freq_semanal_atividade_fisica', 
    'tempo_tecnologia_diario'
]

for col in colunas_para_arredondar:
    df[col] = df[col].round().astype(int)

# FEATURE ENGINEERING ===
df['IMC'] = df['Peso'] / (df['Altura'] ** 2)

coluna_alvo = 'Nivel_Obesidade'

# Mapa de calor 
df_num = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Calor - Correlação")
plt.show()

# 1. LOOP 1 para escolher melhor modelo

print("\n" + "="*60)
print(">>> FASE 1: SELEÇÃO DO MELHOR MODELO COMPLETO")
print("="*60)

features_num_full = ['Idade', 'cons_vegetais_freq', 'num_refeicoes_princ', 'cons_agua_diaria', 'freq_semanal_atividade_fisica', 'tempo_tecnologia_diario', 'IMC']

features_cat_full = ['Genero', 'historico_familiar', 'cons_caloricos_freq', 'cons_lanches_entre_refeicoes', 'fuma', 'monitora_calorias_diarias', 'bebe_alcool', 'meio_transporte_habitual']

X_full = df[features_num_full + features_cat_full]
y = df[coluna_alvo]
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor_scaled_full = ColumnTransformer([
    ('num', StandardScaler(), features_num_full),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat_full)
])

preprocessor_trees_full = ColumnTransformer([
    ('num', 'passthrough', features_num_full),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat_full)
])

models_to_test_full = {
    "Regressao Logistica": (LogisticRegression(max_iter=1000, random_state=42), preprocessor_scaled_full),
    "KNN": (KNeighborsClassifier(n_neighbors=5), preprocessor_scaled_full),
    "LightGBM": (LGBMClassifier(random_state=42, verbose=-1), preprocessor_trees_full),
    "Random Forest": (RandomForestClassifier(random_state=42, max_depth=10), preprocessor_trees_full)
}

melhor_acuracia_full = 0
melhor_modelo_full = None
nome_melhor_full = ""

for name, (classifier, preprocessor) in models_to_test_full.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    pipeline.fit(X_train_full, y_train_full)
    
    # Acurária de treino e teste
    acc_train = accuracy_score(y_train_full, pipeline.predict(X_train_full))
    acc_test = accuracy_score(y_test_full, pipeline.predict(X_test_full))
    
    print(f" [FASE 1] {name} | Treino: {acc_train*100:.2f}% | Teste: {acc_test*100:.2f}%")
    
    if acc_test > melhor_acuracia_full:
        melhor_acuracia_full = acc_test
        melhor_modelo_full = pipeline
        nome_melhor_full = name
    
# 3. EXTRAÇÃO DAS MELHORES VARIÁVEIS (TOP 10 -> SELECIONAR 5)

print("\n" + "-"*60)
print(f">>> ANALISANDO VARIÁVEIS DO VENCEDOR ({nome_melhor_full})")
print("-" * 60)

modelo_para_importancia = melhor_modelo_full

if not hasattr(melhor_modelo_full.named_steps['classifier'], 'feature_importances_'):
    print(" (Usando modelo auxiliar de árvore para ranking de variáveis...)")
    modelo_aux = Pipeline(steps=[
        ('preprocessor', preprocessor_trees_full),
        ('classifier', LGBMClassifier(random_state=42, verbose=-1))
    ])
    modelo_aux.fit(X_train_full, y_train_full)
    modelo_para_importancia = modelo_aux

importances = modelo_para_importancia.named_steps['classifier'].feature_importances_
feature_names_out = modelo_para_importancia.named_steps['preprocessor'].get_feature_names_out()

df_importances = pd.DataFrame({
    'feature': feature_names_out,
    'importance': importances
}).sort_values(by='importance', ascending=False)

df_importances['feature_clean'] = df_importances['feature'].str.replace('num__', '').str.replace('cat__', '')

print("--- TOP 10 FEATURES MAIS IMPORTANTES ---")
print(df_importances[['feature_clean', 'importance']].head(10).to_string(index=False))

cols_to_use_simple = []
todas_colunas = features_num_full + features_cat_full

for feature_transf in df_importances['feature_clean']:
    for col_orig in todas_colunas:
        if feature_transf == col_orig or feature_transf.startswith(col_orig + "_"):
            if col_orig not in cols_to_use_simple:
                cols_to_use_simple.append(col_orig)
    if len(cols_to_use_simple) == 5:
        break

print(f"\n>>> AS 5 VARIÁVEIS ESCOLHIDAS: {cols_to_use_simple}")

# 4. LOOP 2: COM AS 5 VARIÁVEIS

print("\n" + "="*60)
print(">>> FASE 2: RODANDO LOOP COM APENAS AS TOP 5")
print("="*60)

X_simple = df[cols_to_use_simple]
y = df[coluna_alvo]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42, stratify=y
)

features_num_simple = [c for c in cols_to_use_simple if c in features_num_full]
features_cat_simple = [c for c in cols_to_use_simple if c in features_cat_full]

preprocessor_scaled_simple = ColumnTransformer([
    ('num', StandardScaler(), features_num_simple),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat_simple)
])

preprocessor_trees_simple = ColumnTransformer([
    ('num', 'passthrough', features_num_simple),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat_simple)
])

models_to_test_simple = {
    "Regressao Logistica (Top 5)": (LogisticRegression(max_iter=1000, random_state=42), preprocessor_scaled_simple),
    "KNN (Top 5)": (KNeighborsClassifier(n_neighbors=5), preprocessor_scaled_simple),
    "LightGBM (Top 5)": (LGBMClassifier(random_state=42, verbose=-1), preprocessor_trees_simple),
    "Random Forest (Top 5)": (RandomForestClassifier(random_state=42, max_depth=10), preprocessor_trees_simple)
}

melhor_acuracia_final = 0
melhor_modelo_final = None
nome_vencedor_final = ""

for name, (classifier, preprocessor) in models_to_test_simple.items():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    model_pipeline.fit(X_train_s, y_train_s)
    
    # Acurácia de treino e teste
    acc_train = accuracy_score(y_train_s, model_pipeline.predict(X_train_s))
    acc_test = accuracy_score(y_test_s, model_pipeline.predict(X_test_s))
    
    print(f" [FASE 2] {name} | Treino: {acc_train*100:.2f}% | Teste: {acc_test*100:.2f}%")
    
    if acc_test > 0.75:
        print("    -> Status: Aprovado (> 75%)")
    else:
        print("    -> Status: Reprovado")

    if acc_test > melhor_acuracia_final:
        melhor_acuracia_final = acc_test
        melhor_modelo_final = model_pipeline
        nome_vencedor_final = name
 

# ANÁLISE DO MODELO VENCEDOR DA FASE 2 ===
print("\n" + "="*60)
print(f">>> RANKING DAS VARIAVEIS DO MELHOR MODELO FINAL ({nome_vencedor_final})")
print("="*60)

if melhor_modelo_final:
    final_step = melhor_modelo_final.named_steps['classifier']
    preprocessor_step = melhor_modelo_final.named_steps['preprocessor']
    
    # Tentar obter os nomes das features
    try:
        feature_names = preprocessor_step.get_feature_names_out()
    except:
        feature_names = cols_to_use_simple # Fallback se der erro
        
    importancias = None
    
    # 1. Verifica se é modelo baseada em Árvore 
    if hasattr(final_step, 'feature_importances_'):
        importancias = final_step.feature_importances_
        
    # 2. Verifica se é modelo Linear (Regressão Logística) 
    elif hasattr(final_step, 'coef_'):
        importancias = abs(final_step.coef_[0])
        
    # 3. KNN não tem importância de features nativa
    else:
        print("O modelo vencedor (ex: KNN) não possui cálculo nativo de importância de variáveis.")

    # Se conseguiu calcular importância, exibe gráfico e tabela
    if importancias is not None:
        df_final_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importancias
        })
        
        # Limpeza dos nomes (remove prefixos do pipeline)
        df_final_imp['Feature'] = df_final_imp['Feature'].str.replace('num__', '').str.replace('cat__', '')
        
        # Ordenar
        df_final_imp = df_final_imp.sort_values(by='Importance', ascending=False)
        
        # Exibir variáveis que tiverem sobrado)
        print(df_final_imp.head(10).to_string(index=False))
        
        # Gráfico de Barras
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_final_imp.head(10), palette='viridis')
        plt.title(f'Importância das Variáveis - {nome_vencedor_final} (Fase 2)')
        plt.xlabel('Importância Relativa')
        plt.tight_layout()
        plt.show()

# 5. EXPORTAÇÃO

print("\n" + "-"*60)
print(f"MODELO VENCEDOR FINAL: {nome_vencedor_final}")
print(f"   Acurácia Final (Teste): {melhor_acuracia_final*100:.2f}%")
print("-" * 60)

if melhor_modelo_final:
    joblib.dump(melhor_modelo_final, 'modelo_obesidade.pkl')
    print("Modelo salvo com sucesso como 'modelo_obesidade.pkl'")
    print(f"Colunas esperadas pelo modelo: {cols_to_use_simple}")


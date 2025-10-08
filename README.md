# Previsão de Rotatividade de Funcionários (Prediction de Attrition)

> Projeto de análise e modelagem para prever a probabilidade de um funcionário deixar a empresa (attrition) a partir de dados históricos de RH.

![Demonstração do sistema](results/demo-placeholder.png)

## Funcionalidades Principais

- Leitura e limpeza de dados de RH (`data/rh_data.csv`).
- Engenharia de features (ex.: faixas etárias, média de anos por empresa, razão de experiência na empresa).
- Análises exploratórias (distribuições, correlações e tabelas cruzadas por `Attrition`).
- Treinamento e comparação de modelos supervisionados: Regressão Logística, Random Forest e XGBoost.
- Avaliação completa com métricas: Acurácia, Precisão, Recall, F1-Score e AUC-ROC.
- Geração de visualizações de suporte: matrizes de confusão, curvas ROC, importâncias de variáveis e distribuições categóricas.

## Resultados e Conclusões

- Problema: prever se um funcionário sairá (Attrition = 1) a partir de variáveis demográficas, ocupacionais e históricas.
- Principais passos realizados:
  - Tratamento de nulos (preenchimento e remoção controlada).
  - Remoção de colunas constantes (ex.: `EmployeeCount`, `Over18`, `StandardHours`).
  - Criação de variáveis derivadas: `AgeGroup`, `AvgYearsPerCompany`, `CompanyExperienceRatio`, `SalaryHikePerIncome`, `EducationDomain`, entre outras.
  - Remoção de variáveis com baixa contribuição empírica para o modelo (por exemplo: `Education`, `FarFromHome`, `JobLevel`, `NumCompaniesWorked`, `PromotionRate`, `StockOptionLevel`, `TrainingTimesLastYear`, `MonthlyIncome`, `PercentSalaryHike`).
- Modelos testados: Regressão Logística, Random Forest e XGBoost. As métricas avaliadas estão disponíveis no notebook `scripts/analysis.ipynb` (tabela `metrics_df`).
- Interpretação: o notebook calcula coeficientes da regressão logística e importâncias das árvores para identificar as features que mais influenciam a rotatividade. Essas informações ajudam a priorizar ações de retenção (ex.: políticas de carreira, treinamentos ou revisão de benefícios).

> Observação: valores numéricos específicos das métricas (acurácia, AUC etc.) e gráficos gerados estão no notebook `scripts/analysis.ipynb` — recomendo executar o notebook para visualizar os resultados e figuras interativas.

## Tecnologias Utilizadas

- Python 3.12
- Bibliotecas principais: pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib

## Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/vitoriapguimaraes/Python-PredicaoRH.git
cd Python-PredicaoRH
```

2. Instale dependências (recomendo usar um venv):

```bash
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

3. Abra e execute o notebook `scripts/analysis.ipynb` no Jupyter ou VS Code (Execute as células na ordem). O notebook carrega os dados de `data/rh_data.csv` e gera as análises e gráficos.

## Como Usar

- Para reproduzir a análise completa: abra `scripts/analysis.ipynb` e execute todas as células.
- Para treinar somente o modelo em script, você pode extrair as células relevantes do notebook e convertê-las para um script Python (por exemplo `train.py`).
- Explore a aba de métricas (`metrics_df`) e os gráficos (`plot_model_comparison`, `plot_roc_and_confusion`, importâncias) para escolher o modelo a ser usado em produção.

## Estrutura de Diretórios

```
Python-PredicaoRH
├── data/                # Dados brutos (ex.: rh_data.csv)
├── scripts/             # Notebook de análise (scripts/analysis.ipynb)
├── results/             # Sugestão: salvar figuras e relatórios aqui
└── README.md
```

## Status

- ✅ Análise exploratória e modelagem inicial concluídas (notebook).
- 🛠️ Próximos passos: validação cruzada com tuning de hiperparâmetros, calibração de probabilidades, explicabilidade (SHAP) e pipeline de produção.

## Próximos passos recomendados

- Realizar busca de hiperparâmetros (Grid/Random/Optuna) para Random Forest e XGBoost.
- Aplicar validação cruzada estratificada para estimativas mais robustas.
- Calibrar probabilidades (por exemplo `CalibratedClassifierCV`) se o objetivo for usar probabilidade de saída para tomada de decisão.
- Adotar técnicas de interpretabilidade (SHAP/LIME) para explicar previsões a stakeholders de RH.
- Construir um pipeline (pré-processamento + modelo) e empacotar como API REST ou app para consultas em lote.

## Mais Sobre Mim

Acesse os documentos e outros projetos em: https://github.com/vitoriapguimaraes

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Previsor de Rotatividade",
    page_icon="📊",
    layout="wide")

st.title("📊 Previsor de Rotatividade de Funcionários")
st.markdown("Este aplicativo usa **Machine Learning (XGBoost)** para prever a probabilidade de um funcionário deixar a empresa.")
tabs = st.tabs([
    "🎯 Previsão",
    "ℹ️ Métricas do modelo",
    "🔬 Feature Importance"
    ])

with tabs[0]:

    main_column1, spacer, main_column2 = st.columns([3,0.2,1])

    with main_column1:
        st.subheader("⚙️ Parâmetros de Entrada")

        st.code("Altere os dados abaixo para obter a previsão")
        col1, spacer, col2, spacer, col3 = st.columns([1,0.2,1,0.2,1])
        with col1:
            st.markdown("##### 📅 Dados Temporais")
            age = st.slider("Idade", 18, 60, 30)
            total_years = st.slider("Anos totais de experiência", 0, 30, 10)
            years_at_company = st.slider("Anos na empresa atual", 0, 20, 5)
            years_with_manager = st.slider("Anos com o mesmo gerente", 0, 10, 3)

        with col2:
            st.markdown("##### 👤 Dados Pessoais")
            marital = st.segmented_control("Estado civil", ["Single", "Married", "Divorced"], default="Single")
            gender = st.segmented_control("Gênero", ["Male", "Female"], default="Male")
            travel = st.segmented_control("Frequência de viagens", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], default="Non-Travel")
            distance = st.slider("Distância da casa", 0, 50, 10)

        with col3:
            st.markdown("##### 💼 Dados Profissionais")
            department = st.segmented_control("Departamento", ["Sales", "Research & Development", "Human Resources"],default="Sales")
            job_role = st.selectbox("Cargo", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])

        input_data = pd.DataFrame({
            'Age': [age],
            'TotalWorkingYears': [total_years],
            'YearsAtCompany': [years_at_company],
            'YearsWithCurrManager': [years_with_manager],
            'MaritalStatus': [marital],
            'BusinessTravel': [travel],
            'Department': [department],
            'Gender': [gender],
            'JobRole': [job_role],
            'DistanceFromHome': [distance],
            'Attrition': ['No']
        })

        bins = [18, 25, 35, 45, 55, 65]
        labels = ['18-25', '26-35', '36-45', '46-55', '56+']
        input_data['AgeGroup'] = pd.cut(input_data['Age'], bins=bins, labels=labels, right=False)
        input_data['FarFromHome'] = (input_data['DistanceFromHome'] > 10).astype(int) 
        input_data['CompanyExperienceRatio'] = input_data['YearsAtCompany'] / (input_data['TotalWorkingYears'] + 1)
        input_data['AvgYearsPerCompany'] = input_data['TotalWorkingYears'] / (input_data['YearsAtCompany'] + 1)
        input_data['SalaryHikePerIncome'] = 0

        input_data_model = input_data.copy()

        try:
            MODEL_PATH = "data/model"
            encoders = joblib.load(f"{MODEL_PATH}/label_encoders.pkl")
            scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")
            num_cols = joblib.load(f"{MODEL_PATH}/num_cols.pkl")
            xgb_model = joblib.load(f"{MODEL_PATH}/xgb_model.pkl")
        except FileNotFoundError as e:
            st.error(f"❌ Erro ao carregar arquivos do modelo: {e}")
            st.stop()

        categorical_cols = input_data_model.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in encoders:
                try:
                    input_data_model[col] = encoders[col].transform(input_data_model[col])
                except ValueError:
                    input_data_model[col] = 0

        numeric_cols_to_scale = [col for col in num_cols if col in input_data_model.columns]
        if numeric_cols_to_scale:
            input_data_model[numeric_cols_to_scale] = scaler.transform(input_data_model[numeric_cols_to_scale])

    with main_column2:
        st.subheader("🔮 Previsão")
        try:
            if hasattr(xgb_model, 'feature_names_in_'):
                for col in xgb_model.feature_names_in_:
                    if col not in input_data_model.columns:
                        input_data_model[col] = 0
                
                input_data_model = input_data_model[xgb_model.feature_names_in_]
            
            prob = xgb_model.predict_proba(input_data_model)[0][1]

            if prob > 0.7:
                delta_color = "inverse"
                delta_text = "Alto Risco"
                st.error("⚠️ **Atenção:** Alta probabilidade de rotatividade! Ações preventivas são fortemente recomendadas.")
            elif prob > 0.4:
                delta_color = "off"
                delta_text = "Risco Moderado"
                st.warning("⚠️ **Alerta:** Risco moderado de rotatividade. Vale a pena revisar os planos de desenvolvimento e engajamento.")
            else:
                delta_color = "normal"
                delta_text = "Baixo Risco"
                st.success("✅ **Estável:** Baixo risco de saída. O funcionário demonstra boa estabilidade.")

            st.metric(
                label="Probabilidade de Saída",
                value=f"{prob*100:.1f}%",
                delta=delta_text,
                delta_color=delta_color
            )
        
        except Exception as e:
            st.error(f"❌ Erro ao fazer previsão: {e}")
            st.exception(e)

with tabs[1]:
    st.subheader("ℹ️ Métricas do modelo")
    try:
        metrics = joblib.load(f"{MODEL_PATH}/model_metrics.pkl")
        conf_matrix = joblib.load(f"{MODEL_PATH}/confusion_matrix.pkl")
                
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.metric(
                label="🎯 Acurácia",
                value=f"{metrics.get('Acurácia', 0):.2%}",
                help="Proporção de previsões corretas"
            )
        with metric_cols[1]:
            st.metric(
                label="🔍 Precisão",
                value=f"{metrics.get('Precisão', 0):.2%}",
                help="Proporção de positivos previstos que são realmente positivos"
            )
        with metric_cols[2]:
            st.metric(
                label="📡 Recall",
                value=f"{metrics.get('Recall', 0):.2%}",
                help="Proporção de positivos reais que foram identificados"
            )
        with metric_cols[3]:
            st.metric(
                label="⚖️ F1-Score",
                value=f"{metrics.get('F1-Score', 0):.2%}",
                help="Média harmônica entre Precisão e Recall"
            )
        with metric_cols[4]:
            st.metric(
                label="📈 AUC-ROC",
                value=f"{metrics.get('AUC-ROC', 0):.2%}" if metrics.get('AUC-ROC') else "N/A",
                help="Área sob a curva ROC - capacidade de discriminação do modelo"
            )
        
        st.subheader("🎲 Matriz de Confusão")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(16, 4))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                linewidths=2,
                linecolor='white',
                square=True,
                ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'}
            )
            ax.set_xlabel('Predição', fontsize=8, fontweight='600', color='#424242')
            ax.set_ylabel('Real', fontsize=8, fontweight='600', color='#424242')
            ax.set_title('Matriz de Confusão', fontsize=12, fontweight='700', 
                        color='#1f1f1f', pad=8)

            ax.set_xticklabels(['Não Sai (0)', 'Sai (1)'], rotation=0, fontsize=9)
            ax.set_yticklabels(['Não Sai (0)', 'Sai (1)'], rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📖 Interpretação")
            
            tn, fp, fn, tp = conf_matrix.ravel()
            
            st.markdown(f"""
            **Verdadeiros Negativos (TN):** {tn}  
            *Funcionários que ficaram e foram previstos corretamente*
            
            **Falsos Positivos (FP):** {fp}  
            *Funcionários que ficaram mas foram previstos como saída*
            
            **Falsos Negativos (FN):** {fn}  
            *Funcionários que saíram mas foram previstos como permanência*
            
            **Verdadeiros Positivos (TP):** {tp}  
            *Funcionários que saíram e foram previstos corretamente*
            """)
        
    except FileNotFoundError:
        st.info("ℹ️ Métricas do modelo não disponíveis. Execute o treinamento para gerar as métricas.")
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar as métricas: {e}")
        st.exception(e)

with tabs[2]:
    st.subheader("🔬 Importância das Variáveis no Modelo")
    st.markdown("Entenda quais fatores mais influenciam a decisão do modelo de prever rotatividade.")
        
    try:
        feature_importance = pd.DataFrame({
            'Feature': xgb_model.feature_names_in_,
            'Importância': xgb_model.feature_importances_
        }).sort_values('Importância', ascending=False).head(15)
        
        feature_importance['Importância (%)'] = (feature_importance['Importância'] / feature_importance['Importância'].sum() * 100).round(2)
        
        col1, col2 = st.columns([2.5, 1.5])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 7))
            bars = ax.barh(
                feature_importance['Feature'],
                feature_importance['Importância (%)'],
                color='#1f77b4',
                edgecolor='#0d47a1',
                linewidth=1.5,
                alpha=0.85
            )
            for i, (bar, val) in enumerate(zip(bars, feature_importance['Importância (%)'])):
                ax.text(
                    val + 0.3,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%',
                    va='center',
                    fontsize=9,
                    color='#424242',
                    fontweight='500'
                )
            ax.set_xlabel('Importância (%)', fontsize=11, fontweight='600', color='#424242')
            ax.set_ylabel('')
            ax.set_title('Variáveis Mais Importantes', fontsize=13, fontweight='700', 
                        color='#1f1f1f', pad=1)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#e0e0e0')
            ax.spines['bottom'].set_color('#e0e0e0')
            ax.tick_params(axis='both', labelsize=10, colors='#424242')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            top_3_importance = feature_importance.head(3)['Importância (%)'].sum()
            st.metric(
                "Top 3 Features",
                f"{top_3_importance:.1f}%",
                help="Soma da importância das 3 principais variáveis"
            )
            st.dataframe(
                feature_importance[['Feature', 'Importância (%)']].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=500
            )
            
    except Exception as e:
        st.error(f"❌ Erro ao gerar gráfico: {e}")
        st.exception(e)
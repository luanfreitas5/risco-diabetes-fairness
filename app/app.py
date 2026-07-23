"""Dashboard Streamlit de triagem de risco de diabetes.

Ferramenta educacional de portfólio: coleta fatores de risco, calcula a
probabilidade de risco com o modelo treinado (calibrado) e exibe o resultado com
um medidor Plotly. **Não** é diagnóstico médico e não armazena dados pessoais.

Execução::

    uv sync --extra app
    make app            # ou: uv run streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.paths import get_paths
from config.settings import get_settings
from constants import columns as col
from constants import labels as lbl
from models.persistence import load_model
from visualization.theme import PALETTE

MODEL_FILE = "model.joblib"


@st.cache_resource
def _load_cached_model(model_path: str):
    """Carrega o modelo uma única vez (cacheado pelo Streamlit).

    Parameters
    ----------
    model_path : str
        Caminho do artefato de modelo.

    Returns
    -------
    tuple
        Pipeline treinado e seus metadados.
    """
    return load_model(Path(model_path))


def _build_gauge(probability: float, threshold: float) -> go.Figure:
    """Constrói um medidor Plotly para a probabilidade de risco.

    Parameters
    ----------
    probability : float
        Probabilidade prevista (0 a 1).
    threshold : float
        Limiar de decisão para sinalizar risco elevado.

    Returns
    -------
    go.Figure
        Figura do medidor.
    """
    color = PALETTE["accent"] if probability >= threshold else PALETTE["positive"]
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probabilidade de risco de diabetes"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "threshold": {
                    "line": {"color": PALETTE["neutral"], "width": 3},
                    "value": threshold * 100,
                },
            },
        )
    )


def _collect_inputs() -> dict[str, float]:
    """Coleta os fatores de risco pela barra lateral.

    Returns
    -------
    dict[str, float]
        Dicionário feature -> valor informado pelo usuário.
    """
    st.sidebar.header("Fatores de risco")
    binary_label = {0.0: "Não", 1.0: "Sim"}

    inputs: dict[str, float] = {}
    inputs["HighBP"] = st.sidebar.selectbox(
        "Pressão alta", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["HighChol"] = st.sidebar.selectbox(
        "Colesterol alto", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["CholCheck"] = st.sidebar.selectbox(
        "Checou colesterol (5 anos)", [0.0, 1.0], index=1, format_func=binary_label.get
    )
    inputs["BMI"] = float(st.sidebar.slider("IMC", 12.0, 70.0, 28.0))
    inputs["Smoker"] = st.sidebar.selectbox("Fumante", [0.0, 1.0], format_func=binary_label.get)
    inputs["Stroke"] = st.sidebar.selectbox("Já teve AVC", [0.0, 1.0], format_func=binary_label.get)
    inputs["HeartDiseaseorAttack"] = st.sidebar.selectbox(
        "Doença/ataque cardíaco", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["PhysActivity"] = st.sidebar.selectbox(
        "Atividade física", [0.0, 1.0], index=1, format_func=binary_label.get
    )
    inputs["Fruits"] = st.sidebar.selectbox(
        "Consome frutas", [0.0, 1.0], index=1, format_func=binary_label.get
    )
    inputs["Veggies"] = st.sidebar.selectbox(
        "Consome vegetais", [0.0, 1.0], index=1, format_func=binary_label.get
    )
    inputs["HvyAlcoholConsump"] = st.sidebar.selectbox(
        "Consumo pesado de álcool", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["AnyHealthcare"] = st.sidebar.selectbox(
        "Tem plano de saúde", [0.0, 1.0], index=1, format_func=binary_label.get
    )
    inputs["NoDocbcCost"] = st.sidebar.selectbox(
        "Sem médico por custo", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["GenHlth"] = float(st.sidebar.slider("Saúde geral (1=excelente, 5=ruim)", 1, 5, 3))
    inputs["MentHlth"] = float(st.sidebar.slider("Dias de saúde mental ruim (30d)", 0, 30, 0))
    inputs["PhysHlth"] = float(st.sidebar.slider("Dias de saúde física ruim (30d)", 0, 30, 0))
    inputs["DiffWalk"] = st.sidebar.selectbox(
        "Dificuldade para caminhar", [0.0, 1.0], format_func=binary_label.get
    )
    inputs["Sex"] = float(st.sidebar.selectbox("Sexo", [0, 1], format_func=lbl.SEX_LABELS.get))
    inputs["Age"] = float(
        st.sidebar.selectbox(
            "Faixa etária", list(lbl.AGE_LABELS), index=8, format_func=lbl.AGE_LABELS.get
        )
    )
    inputs["Education"] = float(
        st.sidebar.selectbox(
            "Escolaridade",
            list(lbl.EDUCATION_LABELS),
            index=5,
            format_func=lbl.EDUCATION_LABELS.get,
        )
    )
    inputs["Income"] = float(
        st.sidebar.selectbox(
            "Renda", list(lbl.INCOME_LABELS), index=6, format_func=lbl.INCOME_LABELS.get
        )
    )
    return inputs


def main() -> None:
    """Renderiza o dashboard de triagem de risco de diabetes."""
    settings = get_settings()
    st.set_page_config(page_title="Triagem de Risco de Diabetes", page_icon="🩺", layout="wide")
    st.title("🩺 Triagem de Risco de Diabetes — IA Responsável")
    st.caption(
        "Ferramenta educacional de portfólio. **Não** substitui avaliação médica "
        "profissional e **não** constitui diagnóstico."
    )

    threshold = settings.evaluation.decision_threshold
    model_path = get_paths().models / MODEL_FILE
    if not model_path.exists():
        st.warning("Modelo não encontrado. Rode `make train` para gerar `models/model.joblib`.")
        st.stop()

    model, metadata = _load_cached_model(str(model_path))
    inputs = _collect_inputs()

    features = pd.DataFrame([inputs])[col.FEATURES]
    probability = float(model.predict_proba(features)[:, 1][0])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_build_gauge(probability, threshold), use_container_width=True)
    with right:
        flag = "🔴 Risco elevado" if probability >= threshold else "🟢 Risco baixo"
        st.metric("Classificação (limiar de triagem)", flag, f"limiar = {threshold:.0%}")
        st.progress(min(probability, 1.0))
        st.write(
            "Interpretação: a probabilidade indica o risco estimado com base nos "
            "fatores informados. Um resultado elevado sugere procurar avaliação clínica."
        )
        if metadata:
            st.caption(f"Modelo: {metadata.get('active_model', 'n/d')}")

    st.divider()
    st.info(
        settings.project.name
        + " — auditoria de justiça disponível em reports/metrics/fairness_report.json"
    )


if __name__ == "__main__":
    main()

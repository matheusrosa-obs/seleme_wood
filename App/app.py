#################################################
import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path
import plotly.express as px

from warnings import filterwarnings
import re
filterwarnings("ignore")

######## Configurando o caminho para a pasta raiz do projeto ########
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p

####### Carregando os dados no cache do Streamlit ########
@st.cache_data
def load_data(file_path: str | Path) -> gpd.GeoDataFrame:
    resolved_path = _resolve_path(file_path)
    data = gpd.read_file(resolved_path)
    return data

geo = load_data("Dados/Processados/eucalipto_sc.geojson")

################################################
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Seleme Wood",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="logo_dark_mini.png"
)

################################################
# ESTRUTURA DA SIDEBAR
################################################
if "pagina" not in st.session_state:
    st.session_state.pagina = "Demanda 1"

st.markdown(
    """
    <style>
    /* Custom sidebar width */
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 300px;
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 18rem;">
        """,
        unsafe_allow_html=True
    )
    if st.button("Consumo de painéis de madeira"):
        st.session_state.pagina = "Demanda 1"
    if st.button("Produtores de eucalipto"):
        st.session_state.pagina = "Demanda 2"

    st.divider()
    st.image("logo_dark.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

################################################
# CONTEÚDO DAS PÁGINAS
################################################

################### PÁGINA 1 ###################
if st.session_state.pagina == "Demanda 1":
    st.header("Página 1")

################### PÁGINA 2 ###################
elif st.session_state.pagina == "Demanda 2":

    geo["lenha_eucalipto"] = geo["lenha_eucalipto"].fillna(0)
    geo["tora_eucalipto"] = geo["tora_eucalipto"].fillna(0)

    st.divider()

    fig = px.scatter_mapbox(
        geo,
        lat="nu_latitude",
        lon="nu_longitude",
        size="lenha_eucalipto",
        color="lenha_eucalipto",
        hover_name="NM_MUN",
        hover_data={
            "lenha_eucalipto": True,
            "tora_eucalipto": True,
            "altitude": True
        },
        center={"lat": -27.2423, "lon": -50.2189},
        zoom=5.8  # Defina um zoom inicial
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_accesstoken="SEU_MAPBOX_TOKEN_AQUI",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        dragmode="zoom"  # Permite zoom com mouse
    )

    # Ativa o scroll zoom no gráfico
    fig.update_layout(
        mapbox=dict(
            uirevision=True
        )
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    with col2:
        st.write("Tabela de produtores de eucalipto")
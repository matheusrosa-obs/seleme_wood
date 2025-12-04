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
    st.image("logo_dark.png", width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

################################################
# CONTEÚDO DAS PÁGINAS
################################################

################### PÁGINA 1 ###################
if st.session_state.pagina == "Demanda 1":
    st.header("Consumidores potenciais de painéis de madeira no Brasil")

    st.markdown("Distribuição das empresas consumidoras de painéis de madeira por porte e CNAE no Brasil.")

    st.divider()

################### PÁGINA 2 ###################
elif st.session_state.pagina == "Demanda 2":
    st.header("Produtores de eucalipto em Santa Catarina")

    st.markdown("Quantidade produzida de lenha e tora de eucalipto (m³) por município.")

    st.divider()

    geo["tora_eucalipto"] = geo["tora_eucalipto"].fillna(0)
    geo["lenha_eucalipto"] = geo["lenha_eucalipto"].fillna(0)

    # Separando os filtros em colunas
    filtro_col1, filtro_col2 = st.columns(2)

    with filtro_col1:
        tipo_produto = st.selectbox(
            "Selecione o tipo de produto:",
            options=["lenha_eucalipto", "tora_eucalipto"],
            format_func=lambda x: "Lenha de Eucalipto" if x == "lenha_eucalipto" else "Tora de Eucalipto"
        )

    with filtro_col2:
        regioes = geo["NM_RGI"].dropna().unique()
        regioes_opcoes = ["Todos"] + sorted(regioes)
        regiao_selecionada = st.selectbox(
            "Selecione a região:",
            options=regioes_opcoes
        )

    # Filtrando o GeoDataFrame
    geo_filtrado = geo.copy()
    if regiao_selecionada != "Todos":
        geo_filtrado = geo_filtrado[geo_filtrado["NM_RGI"] == regiao_selecionada]

    # Atualizando o mapa de acordo com o filtro selecionado
    fig = px.scatter_mapbox(
        geo_filtrado,
        lat="nu_latitude",
        lon="nu_longitude",
        size=tipo_produto,
        color=tipo_produto,
        color_continuous_scale=px.colors.sequential.Blues,
        hover_name="NM_MUN",
        hover_data={
            "lenha_eucalipto": True,
            "tora_eucalipto": True,
            "altitude": True,
            'nu_latitude': False,
            'nu_longitude': False
        },
        center={"lat": -27.6423, "lon": -51.2189},
        zoom=6.2,
        size_max=40
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        dragmode="zoom",
        height=500
    )

    fig.update_layout(
        mapbox=dict(
            uirevision=True
        )
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    with col2:
        tabela = geo_filtrado[["NM_MUN", tipo_produto, "altitude"]].sort_values(by=tipo_produto, ascending=False)
        st.dataframe(tabela, width='stretch', height=500, hide_index=True)
    
    st.divider()

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Pesquisa Produção da Extração Vegetal e Silvicultura - IBGE (2025).</span>",
        unsafe_allow_html=True
    )

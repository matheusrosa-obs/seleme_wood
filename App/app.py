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
def load_geodata(file_path: str | Path) -> gpd.GeoDataFrame:
    resolved_path = _resolve_path(file_path)
    data = gpd.read_file(resolved_path)
    return data

@st.cache_data
def load_data(file_path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_path(file_path)
    data = pd.read_csv(resolved_path)
    return data

geo = load_geodata("Dados/Processados/eucalipto_sc.geojson")

empresas_eucalipto = load_data("Dados/Processados/empresas_eucalipto_sc.csv")

empresas_pinus = load_data("Dados/Processados/empresas_pinus_munic.csv")

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
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 32vh;">
        """,
        unsafe_allow_html=True
    )
    if st.button("Consumo de painéis de madeira"):
        st.session_state.pagina = "Demanda 1"
    if st.button("Produção de eucalipto"):
        st.session_state.pagina = "Demanda 2"
    st.divider()
    st.image("logo_dark.png", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

################################################
# CONTEÚDO DAS PÁGINAS
################################################

################### PÁGINA 1 ###################
if st.session_state.pagina == "Demanda 1":
    st.header("Consumidores potenciais de painéis de madeira no Brasil")

    st.markdown("Distribuição das empresas consumidoras de painéis de madeira por porte e CNAE no Brasil.")

    st.divider()

    filtro_pinus_col1, filtro_pinus_col2, filtro_pinus_col3 = st.columns(3)

    with filtro_pinus_col1:
        estados = empresas_pinus["nm_uf"].dropna().unique()
        estado_selecionado = st.selectbox(
            "Selecione o estado:",
            options=["Todos"] + sorted(estados)
        )

    with filtro_pinus_col2:
        if estado_selecionado != "Todos":
            municipios = empresas_pinus[empresas_pinus["nm_uf"] == estado_selecionado]["nm_municipio_1mai_ca"].dropna().unique()
        else:
            municipios = empresas_pinus["nm_municipio_1mai_ca"].dropna().unique()
        municipio_selecionado = st.selectbox(
            "Selecione o município:",
            options=["Todos"] + sorted(municipios)
        )

    with filtro_pinus_col3:
        setores = empresas_pinus["nm_cnae_fiscal_principal"].dropna().unique()
        setor_selecionado = st.selectbox(
            "Selecione o grupo setor:",
            options=["Todos"] + sorted(setores)
        )

    empresas_pinus_filtradas = empresas_pinus.copy()
    if estado_selecionado != "Todos":
        empresas_pinus_filtradas = empresas_pinus_filtradas[empresas_pinus_filtradas["nm_uf"] == estado_selecionado]
    if municipio_selecionado != "Todos":
        empresas_pinus_filtradas = empresas_pinus_filtradas[empresas_pinus_filtradas["nm_municipio_1mai_ca"] == municipio_selecionado]
    if setor_selecionado != "Todos":
        empresas_pinus_filtradas = empresas_pinus_filtradas[empresas_pinus_filtradas["nm_cnae_fiscal_principal"] == setor_selecionado]

    ###### Coordenadas e zoom para Brasil ######
    center_coords = {"lat": -14.2350, "lon": -51.9253}
    zoom_level = 3.5

    ###### Coordenadas e zoom para UF selecionada ######
    if estado_selecionado != "Todos" and not empresas_pinus_filtradas.empty:
        center_coords = {
            "lat": empresas_pinus_filtradas["nu_latitude"].mean(),
            "lon": empresas_pinus_filtradas["nu_longitude"].mean()
        }
        zoom_level = 6

    # Atualizando o mapa de acordo com o filtro selecionado
    fig = px.scatter_mapbox(
        empresas_pinus_filtradas,
        lat="nu_latitude",
        lon="nu_longitude",
        size='quantidade_empresas',
        color='quantidade_empresas',
        color_continuous_scale=px.colors.sequential.Greens,
        hover_name="nm_municipio_1mai_ca",
        hover_data={
            'quantidade_empresas': True,
            'nu_latitude': False,
            'nu_longitude': False
        },
        center=center_coords,
        zoom=zoom_level,
        size_max=40
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        dragmode="zoom",
        height=700  # Altura aumentada
    )

    fig.update_layout(
        mapbox=dict(
            uirevision=True
        )
    )

    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    with col2:
        tabela_empresas = empresas_pinus_filtradas.rename(columns={
            "nm_municipio_1mai_ca": "Município",
            "nm_uf": "UF",
            "quantidade_empresas": "Quantidade de Empresas"
        })[["Município", "UF", "Quantidade de Empresas"]]

        if setor_selecionado == "Todos":
            tabela_empresas = tabela_empresas.groupby(["Município", "UF"], as_index=False).agg({"Quantidade de Empresas": "sum"})

        st.dataframe(
            tabela_empresas.sort_values(by="Quantidade de Empresas", ascending=False),
            width='stretch',
            height=700,
            hide_index=True
        )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Receita Federal (2025).</span>",
        unsafe_allow_html=True
    )

    st.divider()












################### PÁGINA 2 ###################
elif st.session_state.pagina == "Demanda 2":
    st.header("Produção de eucalipto em Santa Catarina")

    st.markdown("Quantidade produzida de lenha e tora de eucalipto (m³) por município em 2024.")

    st.divider()

    geo["tora_eucalipto"] = geo["tora_eucalipto"].fillna(0)
    geo["lenha_eucalipto"] = geo["lenha_eucalipto"].fillna(0)

    # Separando os filtros em colunas
    filtro_col1, filtro_col2, filtro_col3 = st.columns(3)

    with filtro_col1:
        tipo_produto = st.selectbox(
            "Selecione o tipo de produto:",
            options=["eucalipto", "lenha_eucalipto", "tora_eucalipto"],
            format_func=lambda x: "Eucalipto (lenha e tora)" if x == "eucalipto" else ("Lenha de Eucalipto" if x == "lenha_eucalipto" else "Tora de Eucalipto")
        )

    with filtro_col2:
        regioes = geo["NM_RGI"].dropna().unique()
        regioes_opcoes = ["Todos"] + sorted(regioes)
        regiao_selecionada = st.selectbox(
            "Selecione a região:",
            options=regioes_opcoes
        )

    with filtro_col3:
        altitude_filtro = st.selectbox(
            "Limite de altitude:",
            options=["Todas", "≤ 800 m"]
        )

    # Filtrando o GeoDataFrame
    geo_filtrado = geo.copy()
    if regiao_selecionada != "Todos":
        geo_filtrado = geo_filtrado[geo_filtrado["NM_RGI"] == regiao_selecionada]
    if altitude_filtro == "≤ 800 m":
        geo_filtrado = geo_filtrado[geo_filtrado["altitude"] <= 800]

    # Formata os valores para milhar e exibe as colunas desejadas
    coluna_produto_mil = f"{tipo_produto}_mil"
    tabela = geo_filtrado[["NM_MUN", coluna_produto_mil, "altitude_mil", tipo_produto]].copy()
    tabela = tabela.sort_values(by=tipo_produto, ascending=False)
    tabela = tabela.rename(columns={
        "NM_MUN": "Município",
        coluna_produto_mil: "Produção (m³)",
        "altitude_mil": "Altitude (m)"
    })    

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
            tipo_produto: False,
            f"{tipo_produto}_mil": True,
            "altitude_mil": True,
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
        st.dataframe(
            tabela[["Município", "Produção (m³)", "Altitude (m)"]],
            width='stretch',
            height=500,
            hide_index=True
        )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Pesquisa Produção da Extração Vegetal e Silvicultura - IBGE (2025).</span>",
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader("Empresas produtoras de eucalipto em Santa Catarina")

    filtro_empresas_col1, filtro_empresas_col2 = st.columns(2)

    with filtro_empresas_col1:
        microrregioes = empresas_eucalipto["Microrregião"].dropna().unique()
        microrregiao_selecionada = st.selectbox(
            "Selecione a microrregião:",
            options=["Todos"] + sorted(microrregioes)
        )

    with filtro_empresas_col2:
        if microrregiao_selecionada != "Todos":
            municipios = empresas_eucalipto[empresas_eucalipto["Microrregião"] == microrregiao_selecionada]["Município"].dropna().unique()
        else:
            municipios = empresas_eucalipto["Município"].dropna().unique()
        municipio_selecionado = st.selectbox(
            "Selecione o município:",
            options=["Todos"] + sorted(municipios)
        )

    empresas_filtradas = empresas_eucalipto.copy()
    if microrregiao_selecionada != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["Microrregião"] == microrregiao_selecionada]
    if municipio_selecionado != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["Município"] == municipio_selecionado]

    st.dataframe(
        empresas_filtradas,
        width='stretch',
        height=400,
        hide_index=True
    )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Receita Federal (2025).</span>",
        unsafe_allow_html=True
    )

    st.divider()
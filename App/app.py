#################################################
import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path
import plotly.express as px
from streamlit_folium import st_folium
import os
from utils import (
    CalculadoraDistanciasAvancada,
    AnalisadorClusters,
    criar_mapa_distancias_cacador,
    obter_rota_ors
)

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
    if str(resolved_path).lower().endswith(('.xlsx', '.xls')):
        data = pd.read_excel(resolved_path)
    else:
        data = pd.read_csv(resolved_path)
    return data

geo = load_geodata("Dados/Processados/eucalipto_sc.geojson")

empresas_eucalipto = load_data("Dados/Processados/empresas_eucalipto_sc.csv")

empresas_pinus = load_data("Dados/Processados/empresas_pinus_munic.csv")

empresas_pinus_tabela = load_data("Dados/Processados/empresas_pinus_tabela.csv")

consumo_painel = load_data("Dados/Processados/consumo_painel.xlsx")

consumo_painel_aberto = load_data("Dados/Processados/consumo_painel_aberto.xlsx")

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
    if st.button("Distâncias e rotas"):
        st.session_state.pagina = "Distâncias Caçador"
    st.divider()
    st.image("logo_dark.png")
    st.markdown("</div>", unsafe_allow_html=True)






################################################
# CONTEÚDO DAS PÁGINAS
################################################

################### PÁGINA 1 ###################
if st.session_state.pagina == "Demanda 1":
    st.header("Consumidores potenciais de painéis de madeira no Brasil")

    st.markdown("Distribuição das empresas consumidoras de painéis de madeira por porte e CNAE no Brasil. Empresas ativas em 2024.")

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
        st.plotly_chart(fig, config={"scrollZoom": True})

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
            height=700,
            hide_index=True
        )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Receita Federal (2024).</span>",
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader("Empresas potenciais consumidoras de painéis no Brasil")

    # Primeira linha de filtros
    filtro_empresas_row1_col1, filtro_empresas_row1_col2 = st.columns(2)
    # Segunda linha de filtros
    filtro_empresas_row2_col1, filtro_empresas_row2_col2 = st.columns(2)

    with filtro_empresas_row1_col1:
        ufs = empresas_pinus_tabela["UF"].dropna().unique()
        uf_selecionada = st.selectbox(
            "Selecione a UF:",
            options=["Todos"] + sorted(ufs),
            key="uf_empresas"
        )

    with filtro_empresas_row1_col2:
        portes = empresas_pinus_tabela["Porte da Empresa"].dropna().unique()
        porte_selecionado = st.selectbox(
            "Selecione o porte:",
            options=["Todos"] + sorted(portes),
            key="porte_empresas"
        )

    with filtro_empresas_row2_col1:
        if uf_selecionada != "Todos":
            municipios = empresas_pinus_tabela[empresas_pinus_tabela["UF"] == uf_selecionada]["Município"].dropna().unique()
        else:
            municipios = empresas_pinus_tabela["Município"].dropna().unique()
        municipio_selecionado = st.selectbox(
            "Selecione o município:",
            options=["Todos"] + sorted(municipios),
            key="municipio_empresas"
        )

    with filtro_empresas_row2_col2:
        setores = empresas_pinus_tabela["CNAE Principal"].dropna().unique()
        setor_selecionado = st.selectbox(
            "Selecione o setor (CNAE Principal):",
            options=["Todos"] + sorted(setores),
            key="setor_empresas"
        )

    empresas_filtradas = empresas_pinus_tabela.copy()
    if uf_selecionada != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["UF"] == uf_selecionada]
    if porte_selecionado != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["Porte da Empresa"] == porte_selecionado]
    if municipio_selecionado != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["Município"] == municipio_selecionado]
    if setor_selecionado != "Todos":
        empresas_filtradas = empresas_filtradas[empresas_filtradas["CNAE Principal"] == setor_selecionado]

    st.dataframe(
        empresas_filtradas[
            [
                "Razão Social",
                "Porte da Empresa",
                "Ano de Início",
                "UF",
                "Microrregião",
                "Município",
                "Endereço Completo",
                "CNAE Principal"
            ]
        ],
        height=400,
        hide_index=True
    )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Receita Federal (2024).</span>",
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Produção e consumo de painéis de madeira compensada (1997-2023)")
        fig1 = px.line(
            consumo_painel,
            x="Ano",
            y=consumo_painel.columns[-3:],
            markers=True,
            labels={"Ano": "Ano", "value": "Consumo (m³)", "variable": "Tipo de Painel"},
            title="Brasil"
        )
        fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Consumo aparente por tipo de painel (1997-2023)")
        consumo_painel_aberto_long = consumo_painel_aberto.melt(
            id_vars="Ano",
            value_vars=consumo_painel_aberto.columns[-3:],
            var_name="Tipo de Painel",
            value_name="Consumo (m³)"
        )
        fig2 = px.line(
            consumo_painel_aberto_long,
            x="Ano",
            y="Consumo (m³)",
            color="Tipo de Painel",
            markers=True,
            labels={"Ano": "Ano", "Consumo (m³)": "Consumo (m³)", "Tipo de Painel": "Tipo de Painel"},
            title="Brasil"
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            yaxis=dict(
                dtick=4000000,
                gridcolor="rgba(200,200,200,0.3)"
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: MDIC (2024), IBGE (2024).</span>",
        unsafe_allow_html=True
    )








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

    # Atualizando o mapa para exibir polígonos dos municípios ao invés de pontos
    fig = px.choropleth_mapbox(
        geo_filtrado,
        geojson=geo_filtrado.geometry,
        locations=geo_filtrado.index,
        color=tipo_produto,
        color_continuous_scale=px.colors.sequential.YlOrRd,
        hover_name="NM_MUN",
        hover_data={
            tipo_produto: False,
            f"{tipo_produto}_mil": True,
            "altitude_mil": True
        },
        center={"lat": -27.6423, "lon": -51.2189},
        zoom=6.2
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
        st.plotly_chart(fig, config={"scrollZoom": True})

    with col2:
        st.dataframe(
            tabela[["Município", "Produção (m³)", "Altitude (m)"]],
            height=500,
            hide_index=True
        )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Pesquisa Produção da Extração Vegetal e Silvicultura - IBGE (2024).</span>",
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
        empresas_filtradas[
            [
                "Razão Social",
                "Porte da Empresa",
                "Ano de Início",
                "UF",
                "Microrregião",
                "Município",
                "Endereço Completo",
                "CNAE Principal"
            ]],
        height=400,
        hide_index=True
    )

    st.markdown(
        "<span style='font-size: 0.85em;'>Fonte: Receita Federal (2024).</span>",
        unsafe_allow_html=True
    )

    st.divider()



################### PÁGINA 3 ###################
elif st.session_state.pagina == "Distâncias Caçador":
    municipio_ref = {
        'cd_mun_ibge': '4203006',
        'nm_mun': 'Caçador',
        'sg_uf': 'SC',
        'latitude': -26.790294,
        'longitude': -51.000398
    }

    # Tenta pegar do st.secrets; se não existir, tenta variável de ambiente
    ORS_API_KEY = None
    try:
        ORS_API_KEY = st.secrets["ors"]["api_key"]
    except Exception:
        ORS_API_KEY = os.getenv("ORS_API_KEY", None)

    # Header
    st.title("📍 Análise Geoespacial de Distâncias")
    st.markdown("**Referência fixa:** Município de **Caçador/SC** | Distâncias calculadas entre Caçador e todas as empresas da base")

    # Carregamento automático de dados
    @st.cache_data(show_spinner="📂 Carregando dados...")
    def carregar_dados():
        arquivo_csv = Path(__file__).parent / "Empresas_Cnae_Geo.csv"
        if arquivo_csv.exists():
            df = pd.read_csv(arquivo_csv, dtype={'cd_mun_ibge': str})
            return df
        else:
            return None

    df_geo = carregar_dados()

    if df_geo is None:
        st.error("❌ Arquivo Empresas_Cnae_Geo.csv não encontrado!")
        st.stop()

    # Botão principal no topo
    col_btn, col_info = st.columns([1, 4])

    with col_btn:
        if st.button("🚀 Processar", type="primary", use_container_width=True):
            with st.spinner("⏳ Processando..."):
                # Cálculo de distâncias
                calc = CalculadoraDistanciasAvancada(df_geo, municipio_ref)
                df_dist = calc.calcular_todas_distancias()
                kpis = calc.extrair_kpis_completos(df_dist)

                # Clustering
                analisador = AnalisadorClusters(df_dist)
                df_cluster = analisador.analise_completa(raio_dbscan=100)

                # Salvar em session_state
                st.session_state['df_dist'] = df_cluster
                st.session_state['kpis'] = kpis
                st.session_state['processado'] = True

                st.success("✅ Processamento concluído!")
                st.rerun()

    with col_info:
        st.info(f"📍 Total de registros: **{len(df_geo):,}**")

    st.markdown("---")

    # Área principal
    if 'processado' not in st.session_state:
        st.info("👈 Clique em **🚀 Processar** para calcular distâncias e clusters")

    else:
        df_dist = st.session_state['df_dist']
        kpis = st.session_state['kpis']

        # KPIs principais
        st.markdown("### 📊 KPIs Principais")

        col1, col2, col3, col4 = st.columns(4)

        mp = kpis['mais_proximo']
        md = kpis['mais_distante']
        est = kpis['estatisticas']
        dist = kpis['distribuicao']

        # Função helper para tratar strings vazias/NaN
        def safe_str(value, max_len=30, default="(Sem nome)"):
            if pd.isna(value) or value == "" or not isinstance(value, str):
                return default
            return value[:max_len] + ("..." if len(value) > max_len else "")

        with col1:
            st.metric("🟢 Mais Próximo", f"{mp['distancia_km']:.1f} km", safe_str(mp['nome_fantasia']))

        with col2:
            st.metric("🔴 Mais Distante", f"{md['distancia_km']:.1f} km", safe_str(md['nome_fantasia']))

        with col3:
            st.metric("📏 Distância Média", f"{est['media_km']:.1f} km", f"Med: {est['mediana_km']:.1f} km")

        with col4:
            st.metric("📍 Total de Pontos", f"{len(df_dist):,}")

        st.markdown("---")

        # Seleção de empresa
        st.markdown("### 📌 Seleção de Empresa para Destaque")

        col_sel1, col_sel2 = st.columns([3, 1])

        with col_sel1:
            nomes_fantasia = (
                df_dist['nm_nome_fantasia']
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            nm_fantasia_sel = st.selectbox(
                "🏢 Selecione uma empresa para destacar no mapa e traçar a rota:",
                options=["(Nenhuma)"] + nomes_fantasia,
                index=0,
                help="A empresa selecionada será destacada em azul no mapa com rota traçada"
            )

            if nm_fantasia_sel == "(Nenhuma)":
                nm_fantasia_sel = None

        with col_sel2:
            if nm_fantasia_sel:
                empresa_info = df_dist[df_dist['nm_nome_fantasia'] == nm_fantasia_sel].iloc[0]
                
                # Obter distância viária da API
                dist_viaria = None
                if ORS_API_KEY:
                    coord_ref = (municipio_ref['latitude'], municipio_ref['longitude'])
                    coord_dest = (empresa_info['latitude'], empresa_info['longitude'])
                    rota = obter_rota_ors(coord_ref, coord_dest, api_key=ORS_API_KEY)
                    if rota:
                        dist_viaria = rota['distance_km']
                
                # Exibir distância reta
                st.metric(
                    "Distância Reta",
                    f"{empresa_info['distancia_km']:.2f} km",
                    f"{empresa_info['nm_mun']}/{empresa_info['sg_uf']}"
                )
                
                # Exibir distância viária se disponível
                if dist_viaria:
                    diferenca = dist_viaria - empresa_info['distancia_km']
                    st.metric(
                        "Distância Viária",
                        f"{dist_viaria:.2f} km",
                        f"+{diferenca:.2f} km" if diferenca > 0 else f"{diferenca:.2f} km"
                    )

        # Mapa
        st.markdown("### 🗺️ Mapa Interativo de Distâncias")

        with st.spinner("🗺️ Gerando mapa..."):
            mapa = criar_mapa_distancias_cacador(
                df_distancias=df_dist,
                municipio_ref=municipio_ref,
                kpis=kpis,
                nm_fantasia_selecionado=nm_fantasia_sel,
                ors_api_key=ORS_API_KEY
            )

        st_folium(mapa, width=1200, height=650, returned_objects=[])

        st.markdown("---")

        # Tabela detalhada
        st.markdown("### 📋 Tabela Detalhada")

        df_view = df_dist.sort_values('distancia_km').copy()

        # Destacar empresa selecionada na tabela
        if nm_fantasia_sel:
            mask_sel = df_view['nm_nome_fantasia'] == nm_fantasia_sel
            df_view = pd.concat([df_view[mask_sel], df_view[~mask_sel]])

        cols_mostrar = [
            'nm_nome_fantasia', 'nm_razao_social',
            'nm_mun', 'sg_uf', 'distancia_km',
            'nm_porte_obs', 'cd_cnae_fiscal_principal',
            'nm_cnae_fiscal_principal', 'cluster_dbscan', 'cluster_kmeans'
        ]
        cols_mostrar = [c for c in cols_mostrar if c in df_view.columns]

        st.dataframe(
            df_view[cols_mostrar].reset_index(drop=True),
            use_container_width=True,
            height=400
        )

        # Análise de clusters
        st.markdown("### 🔍 Análise de Clusters (DBSCAN)")

        if 'cluster_dbscan' in df_view.columns:
            clusters_validos = sorted([c for c in df_view['cluster_dbscan'].unique() if c != -1])

            if clusters_validos:
                for c_id in clusters_validos:
                    df_c = df_view[df_view['cluster_dbscan'] == c_id]
                    dist_media = df_c['distancia_km'].mean()
                    muni_top = df_c['nm_mun'].value_counts().head(3)

                    with st.expander(f"🔵 Cluster {c_id} - {len(df_c)} empresas"):
                        col_c1, col_c2, col_c3 = st.columns(3)
                        col_c1.metric("Empresas", len(df_c))
                        col_c2.metric("Distância Média", f"{dist_media:.1f} km")
                        col_c3.metric("Municípios", df_c['nm_mun'].nunique())

                        st.markdown("**Principais municípios:**")
                        for mun, count in muni_top.items():
                            st.write(f"- {mun}: {count} empresas")
            else:
                st.info("ℹ️ Nenhum cluster denso identificado com os parâmetros atuais (raio 100 km)")

        # Download
        st.markdown("---")
        st.markdown("### 💾 Download dos Dados")

        csv = df_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar tabela completa (CSV)",
            data=csv,
            file_name="analise_distancias_cacador.csv",
            mime="text/csv",
            use_container_width=True
        )
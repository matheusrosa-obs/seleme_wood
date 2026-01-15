import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px


# =========================
# Helpers de caminho
# =========================
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p


@st.cache_data
def load_data(file_path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_path(file_path)
    if str(resolved_path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(resolved_path)
    return pd.read_csv(resolved_path)


# =========================
# Estilo e config
# =========================
try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.set_page_config(
    page_title="Seleme Wood",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="logo_dark_mini.png",
)

# Sidebar width (opcional)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 300px;
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Sidebar / Navegação
# =========================
if "pagina" not in st.session_state:
    st.session_state.pagina = "Empresas por Canal"

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 32vh;">
        """,
        unsafe_allow_html=True,
    )

    if st.button("Empresas por Canal"):
        st.session_state.pagina = "Empresas por Canal"

    st.divider()
    try:
        st.image("logo_dark.png")
    except Exception:
        pass
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ÚNICA PÁGINA: Empresas por Canal
# =========================
if st.session_state.pagina == "Empresas por Canal":
    st.header("Empresas por Canal: E-commerces & Portfólio")

    tab_ecom, tab_port = st.tabs(["E-commerces", "Portfólio"])

    # Placeholder para as duas tabs
    placeholder_df = pd.DataFrame(
        {
            "Município": ["Município A", "Município B", "Município C"],
            "UF": ["SC", "PR", "RS"],
            "Quantidade": [12, 5, 3],
        }
    ).assign(
        nu_latitude=[-27.6, -26.9, -29.2],
        nu_longitude=[-48.6, -49.3, -51.2],
    )

    # ---------- TAB: E-commerces (placeholder) ----------
    with tab_ecom:
        st.markdown(
            "Breve explicação: nesta aba colocaremos os e-commerces que atuam como consumidores ou revendedores. (Placeholder)"
        )
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Tabela (placeholder)**")
            st.dataframe(placeholder_df[["Município", "UF", "Quantidade"]], height=300, hide_index=True)

        with c2:
            st.markdown("**Mapa (placeholder)**")
            fig_e = px.scatter_mapbox(
                placeholder_df,
                lat="nu_latitude",
                lon="nu_longitude",
                size="Quantidade",
                color="Quantidade",
                hover_name="Município",
                center={"lat": -27.0, "lon": -50.0},
                zoom=5,
                size_max=30,
            )
            fig_e.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
            )
            st.plotly_chart(fig_e, use_container_width=True)

    # ---------- TAB: Portfólio ----------
    with tab_port:
        st.markdown("Breve explicação: empresas do portfólio — foco em quem trabalha com painéis de pinus.")

        # Carregar dados
        try:
            df_companies = load_data("App/companies.csv")
            df_items = load_data("App/items.csv")
        except Exception as e:
            st.error(f"Erro ao carregar os arquivos CSV: {e}")
            st.stop()

        # Normalizações simples
        if "nm_empresa" in df_companies.columns:
            df_companies["nm_empresa"] = df_companies["nm_empresa"].astype(str).str.strip()
        if "nm_empresa" in df_items.columns:
            df_items["nm_empresa"] = df_items["nm_empresa"].astype(str).str.strip()

        # Filtrar empresas relevantes para painel de pinus
        works_flag = "works_with_pine_panel" in df_companies.columns
        pine_flag = "pine_panel_flag" in df_companies.columns
        mentions_pinus = "item_count_mentions_pinus" in df_companies.columns

        mask_pine = pd.Series(False, index=df_companies.index)

        if works_flag:
            mask_pine = mask_pine | df_companies["works_with_pine_panel"].astype(str).str.lower().eq("yes")
        if pine_flag:
            mask_pine = mask_pine | (df_companies["pine_panel_flag"].fillna(0).astype(int) == 1)
        if mentions_pinus:
            mask_pine = mask_pine | (df_companies["item_count_mentions_pinus"].fillna(0).astype(int) > 0)

        df_portfolio = df_companies[mask_pine].copy()

        # KPIs
        total_empresas = len(df_portfolio)
        if "product_name" in df_items.columns and "nm_empresa" in df_items.columns:
            total_produtos = int(
                df_items[df_items["nm_empresa"].isin(df_portfolio.get("nm_empresa", []))]["product_name"].nunique()
            )
        else:
            total_produtos = 0

        k1, k2 = st.columns(2)
        k1.metric("Empresas com indícios de painel de pinus", f"{total_empresas}")
        k2.metric("Produtos únicos (empresas filtradas)", f"{total_produtos}")

        st.divider()

        # Filtros + tabela
        fcol1, fcol2 = st.columns([2, 1])

        with fcol1:
            if "status" in df_portfolio.columns:
                status_opts = ["Todos"] + sorted(df_portfolio["status"].dropna().unique().astype(str).tolist())
            else:
                status_opts = ["Todos"]
            status_sel = st.selectbox("Filtrar por status:", status_opts, index=0)

        with fcol2:
            busca = st.text_input("Buscar empresa (nome):")

        df_view = df_portfolio.copy()
        if status_sel != "Todos" and "status" in df_view.columns:
            df_view = df_view[df_view["status"].astype(str) == status_sel]
        if busca and "nm_empresa" in df_view.columns:
            df_view = df_view[df_view["nm_empresa"].str.contains(busca, case=False, na=False)]

        cols_show = [
            c
            for c in [
                "nm_empresa",
                "website_url",
                "status",
                "works_with_wood",
                "works_with_pine_panel",
                "confidence_pine_panel",
                "item_count_total",
                "item_count_wood_related",
                "item_count_mentions_pinus",
            ]
            if c in df_view.columns
        ]

        if cols_show:
            st.dataframe(
                df_view[cols_show].rename(
                    columns={
                        "nm_empresa": "Empresa",
                        "website_url": "Website",
                        "works_with_wood": "Works with wood",
                        "works_with_pine_panel": "Works with pine panel",
                    }
                ),
                height=300,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Nenhuma coluna esperada encontrada para exibir a tabela do portfólio.")

        st.markdown("**Detalhe do portfólio por empresa**")

        empresas_opts = df_view["nm_empresa"].dropna().tolist() if "nm_empresa" in df_view.columns else []
        empresa_sel = st.selectbox("Selecione uma empresa para ver produtos:", options=["(Nenhuma)"] + empresas_opts)

        if empresa_sel and empresa_sel != "(Nenhuma)" and "nm_empresa" in df_items.columns:
            df_products = df_items[df_items["nm_empresa"] == empresa_sel].copy()
            if df_products.empty:
                st.info("Nenhum produto encontrado para a empresa selecionada.")
            else:
                cols_prod = [
                    c
                    for c in ["product_category", "product_name", "materials", "is_wood_related", "mentions_pinus"]
                    if c in df_products.columns
                ]
                st.dataframe(
                    df_products[cols_prod].rename(
                        columns={
                            "product_category": "Categoria",
                            "product_name": "Produto",
                            "materials": "Materiais",
                            "is_wood_related": "Wood-related",
                            "mentions_pinus": "Menção a Pinus",
                        }
                    ),
                    height=300,
                    hide_index=True,
                    use_container_width=True,
                )

        st.markdown("**Mapa (placeholder)**")
        fig_p = px.scatter_mapbox(
            placeholder_df,
            lat="nu_latitude",
            lon="nu_longitude",
            size="Quantidade",
            color="UF",
            hover_name="Município",
            center={"lat": -27.0, "lon": -50.0},
            zoom=5,
            size_max=30,
        )
        fig_p.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
        )
        st.plotly_chart(fig_p, use_container_width=True)

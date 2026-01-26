import streamlit as st
from pathlib import Path
import pandas as pd

from tabs.tab_ecom import render_tab_ecom
from tabs.tab_portfolio import render_tab_portfolio


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p


@st.cache_data
def load_data(file_path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {resolved_path}")

    suf = resolved_path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(resolved_path)

    # CSV: tenta , e depois ;
    try:
        return pd.read_csv(resolved_path)
    except Exception:
        return pd.read_csv(resolved_path, sep=";")


# Estilo e config
try:
    with open(_resolve_path("App/style.css"), encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.set_page_config(
    page_title="Seleme Wood",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(_resolve_path("App/logo_dark_mini.png")),
)

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

# Navegação
if "pagina" not in st.session_state:
    st.session_state.pagina = "Empresas por Canal"

with st.sidebar:
    if st.button("Empresas por Canal"):
        st.session_state.pagina = "Empresas por Canal"
    st.divider()
    try:
        st.image(str(_resolve_path("App/logo_dark.png")))
    except Exception:
        pass

# Página
if st.session_state.pagina == "Empresas por Canal":
    st.header("Empresas por Canal: E-commerces & Portfólio")

    tab_ecom, tab_port = st.tabs(["E-commerces", "Portfólio"])

    # Placeholder só para mapa na aba 2 (opcional)
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

    # ✅ Aba 1 independente: NÃO passa placeholder_df (senão quebra as colunas)
    with tab_ecom:
        render_tab_ecom()

    # Aba 2 independente: usa companies/items
    with tab_port:
        render_tab_portfolio(
            load_data_fn=load_data,
            placeholder_df=None,
            companies_path="App/companies_final.csv",
            items_path="App/items_final.csv",
        )

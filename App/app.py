import streamlit as st
import pandas as pd
from pathlib import Path

from tabs.tab_ecom import render_tab_ecom
from tabs.tab_portfolio import render_tab_portfolio


# =========================
# Helpers de caminho
# =========================
def _project_root() -> Path:
    # app.py está em App/app.py -> parent.parent volta para a raiz do projeto
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
        st.image(str(_resolve_path("App/logo_dark.png")))
    except Exception:
        pass
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Página: Empresas por Canal
# =========================
if st.session_state.pagina == "Empresas por Canal":
    st.header("Empresas por Canal: E-commerces & Portfólio")

    tab_ecom, tab_port = st.tabs(["E-commerces", "Portfólio"])

    # Placeholder compartilhado entre tabs (mantido do seu script)
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

    with tab_ecom:
        render_tab_ecom(placeholder_df=placeholder_df)

    with tab_port:
        render_tab_portfolio(
            load_data_fn=load_data,
            placeholder_df=placeholder_df,
            companies_path="App/companies.csv",
            items_path="App/items.csv",
        )

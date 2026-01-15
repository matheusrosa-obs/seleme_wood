from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin

import streamlit as st
import plotly.express as px
import pandas as pd


CSV_PATH = r"App\lista_produtos_ecommerce_SC_all_sites_exploded (2).csv"


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Dados").exists():
            return parent
    return here.parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p


@st.cache_data
def load_data(file_path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_path(file_path)
    if str(resolved_path).lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(resolved_path)
    else:
        data = pd.read_csv(resolved_path)
    return data


def _safe_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def _normalize_col_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = { _normalize_col_name(c): c for c in df.columns }
    for candidate in candidates:
        key = _normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def render_tab_ecom(
    placeholder_df: pd.DataFrame | None = None,
    csv_path: str | Path = CSV_PATH,
) -> None:
    st.markdown(
        "Breve explicacao: nesta aba colocaremos os e-commerces que atuam como consumidores ou revendedores."
    )

    df = placeholder_df
    if df is None:
        try:
            df = load_data(csv_path)
        except FileNotFoundError:
            st.error(f"Arquivo nao encontrado: {csv_path}")
            return

    if df.empty:
        st.warning("Nenhum dado disponivel.")
        return

    # Default filter
    if "status" in df.columns:
        df = df[df["status"] == "FOUND"]

    # KPIs and map data (enterprise-level view)
    nm_col = "nm_empresa" if "nm_empresa" in df.columns else None
    ppm3_col = "price_per_m3" if "price_per_m3" in df.columns else None

    enterprise_count = int(df[nm_col].nunique()) if nm_col else 0
    product_count = int(df.shape[0])
    median_ppm3 = round(float(df[ppm3_col].median()), 2) if ppm3_col and df[ppm3_col].notna().any() else None

    st.markdown("**Indicadores**")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Empresas", enterprise_count)
    with k2:
        st.metric("Produtos", product_count)
    with k3:
        st.metric("Preco m3 (mediana)", median_ppm3 if median_ppm3 is not None else "-")

    lat_col = _first_existing_column(df, ["nu_latitude", "latitude", "lat"])
    lon_col = _first_existing_column(df, ["nu_longitude", "longitude", "lon", "nu_long"])
    if lat_col and lon_col and nm_col:
        city_col = _first_existing_column(
            df, ["cidade", "municipio", "municipio_nome", "cidade_nome", "nm_municipio", "nm_mun"]
        )
        df = df.copy()
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        company_summary = (
            df.groupby(nm_col, dropna=False)
            .agg(
                nu_latitude=(lat_col, "first"),
                nu_longitude=(lon_col, "first"),
            )
            .reset_index()
        )

        map_df = company_summary.dropna(subset=["nu_latitude", "nu_longitude"]).copy()
        map_df = (
            map_df.groupby(["nu_latitude", "nu_longitude"], dropna=False)[nm_col]
            .apply(lambda s: sorted({v for v in s if pd.notna(v)}))
            .reset_index()
        )
        map_df["empresas_count"] = map_df[nm_col].apply(len)
        map_df["empresas_str"] = map_df[nm_col].apply(
            lambda vals: ", ".join(vals[:8]) + ("..." if len(vals) > 8 else "")
        )

        st.markdown("**Mapa**")
        fig_e = px.scatter_mapbox(
            map_df,
            lat="nu_latitude",
            lon="nu_longitude",
            size="empresas_count",
            color="empresas_count",
            hover_data={
                "empresas_str": True,
                "empresas_count": True,
                "nu_latitude": False,
                "nu_longitude": False,
            },
            size_max=30,
        )
        fig_e.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": -14.2, "lon": -51.9},
            mapbox_zoom=3.2,
            margin=dict(l=0, r=0, t=0, b=0),
            height=380,
        )
        fig_e.update_traces(
            marker=dict(opacity=0.85, sizemin=6),
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
        st.plotly_chart(fig_e, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("Mapa indisponivel: faltam latitude/longitude ou nm_empresa.")

    # Filters and table (product-level view)
    st.markdown("**Filtros**")
    filtered = df.copy()

    city_col = _first_existing_column(
        df, ["cidade", "municipio", "municipio_nome", "cidade_nome", "nm_municipio", "nm_mun"]
    )
    if city_col:
        city_options = sorted(_safe_series(df[city_col]).unique())
        selected_cities = st.multiselect("Cidade", options=city_options)
        if selected_cities:
            filtered = filtered[_safe_series(filtered[city_col]).isin(selected_cities)]
    else:
        st.info("Coluna de cidade nao encontrada. Adicione-a no merge.")

    if nm_col:
        company_options = sorted(_safe_series(filtered[nm_col]).unique())
        selected_companies = st.multiselect("Empresa", options=company_options)
        if selected_companies:
            filtered = filtered[_safe_series(filtered[nm_col]).isin(selected_companies)]

    st.markdown("**Produtos (detalhe)**")
    display_df = filtered.copy()
    if "website_url" in display_df.columns and "url" in display_df.columns:
        base_urls = _safe_series(display_df["website_url"]).str.strip()
        product_urls = _safe_series(display_df["url"]).str.strip()
        display_df["product_url"] = [
            urljoin(base, path) if path else base
            for base, path in zip(base_urls, product_urls)
        ]

    dims_cols = ["length_mm", "width_mm", "thickness_mm"]
    derived_cols = ["price_per_m3", "volume_m3"]
    label_missing = "Dimensoes nao disponiveis"
    if all(col in display_df.columns for col in dims_cols):
        missing_dims = display_df[dims_cols].isna().any(axis=1)
        for col in dims_cols + derived_cols:
            if col in display_df.columns:
                display_df.loc[missing_dims, col] = label_missing

    desired_cols = [
        "nm_empresa",
        "website_url",
        city_col if city_col else None,
        "name",
        "dimensions_raw",
        "price_value",
        "price_per_m3",
        "length_mm",
        "width_mm",
        "thickness_mm",
        "product_url",
    ]
    table_cols = [col for col in desired_cols if col and col in filtered.columns]
    st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)

    if "product_url" in display_df.columns:
        st.markdown("**Preview do produto**")
        preview_options = display_df["product_url"].dropna().unique().tolist()
        if preview_options:
            selected_url = st.selectbox("Selecionar URL para visualizar", preview_options)
            if selected_url:
                st.components.v1.iframe(selected_url, height=520, scrolling=True)
        else:
            st.info("Nao ha URLs de produto disponiveis para preview.")

from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin

import streamlit as st
import pandas as pd


CSV_PATH = r"App/lista_produtos_ecommerce_SC_all_sites_exploded (3).csv"


# =========================
# Infra / IO
# =========================
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
        return pd.read_excel(resolved_path)
    return pd.read_csv(resolved_path)


# =========================
# Utils
# =========================
def _safe_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def _to_float_series(s: pd.Series) -> pd.Series:
    # aceita vírgula decimal
    ss = _safe_series(s).str.replace(",", ".", regex=False)
    return pd.to_numeric(ss, errors="coerce")


def _normalize_col_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_normalize_col_name(c): c for c in df.columns}
    for candidate in candidates:
        key = _normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _pretty_multiline(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "—"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return "\n".join([f"- {ln.lstrip('-• ').strip()}" for ln in lines])
    return t


def _inject_cards_css_once() -> None:
    st.markdown(
        """
        <style>
        .cardx {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 14px 14px;
            background: rgba(20,20,25,0.35);
            margin-bottom: 12px;
        }
        .cardx h4 { margin: 0 0 6px 0; }
        .cardx .meta { font-size: 0.9rem; opacity: 0.9; margin-bottom: 10px; }
        .pill {
            display:inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 0.85rem;
            margin-right: 6px;
            margin-bottom: 6px;
            opacity: 0.95;
        }
        .mono { white-space: pre-wrap; line-height: 1.35; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_empresa(nome: str, url: str, resumo: str, kpis: dict) -> None:
    resumo_fmt = _pretty_multiline(resumo)

    url_html = (
        f"<a href='{url}' target='_blank' style='text-decoration: underline; opacity:0.95;'>{url}</a>"
        if url
        else "<span style='opacity:0.75'>—</span>"
    )

    pills = [f"<span class='pill'>{label}: <b>{val}</b></span>" for label, val in kpis.items()]
    pills_html = " ".join(pills)

    st.markdown(
        f"""
        <div class="cardx">
            <h4>{nome}</h4>
            <div class="meta">{url_html}</div>
            <div>{pills_html}</div>
            <div class="mono" style="margin-top:10px;">{resumo_fmt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Tab ECOM (v2) - UX igual portfolio, sem mapa
# =========================
def render_tab_ecom(
    placeholder_df: pd.DataFrame | None = None,
    csv_path: str | Path = CSV_PATH,
) -> None:
    st.markdown("### E-commerces")
    st.caption("UX no estilo Portfólio: filtros globais, cards de empresas e tabela de produtos. (Sem mapa)")

    _inject_cards_css_once()

    df = placeholder_df
    if df is None:
        try:
            df = load_data(csv_path)
        except FileNotFoundError:
            st.error(f"Arquivo não encontrado: {csv_path}")
            return

    if df.empty:
        st.warning("Nenhum dado disponível.")
        return

    # Colunas principais
    nm_col = _first_existing_column(df, ["nm_empresa", "nome_empresa", "empresa"])
    city_col = _first_existing_column(df, ["nm_mun", "cidade", "municipio"])
    uf_col = _first_existing_column(df, ["nm_uf", "uf"])
    status_col = _first_existing_column(df, ["status"])
    site_col = _first_existing_column(df, ["website_url", "url_site", "site"])
    msg_col = _first_existing_column(df, ["message_pt", "mensagem_pt", "mensagem", "message"])
    ppm3_col = _first_existing_column(df, ["price_per_m3"])
    prod_name_col = _first_existing_column(df, ["name", "nome_produto", "product_name"])
    prod_url_col = _first_existing_column(df, ["product_url", "url"])
    price_col = _first_existing_column(df, ["price_value", "price"])
    dims_raw_col = _first_existing_column(df, ["dimensions_raw", "dimensoes_raw"])

    if not nm_col:
        st.error("Não encontrei a coluna da empresa (ex.: nm_empresa).")
        return

    df = df.copy()
    df[nm_col] = _safe_series(df[nm_col]).str.strip()

    if site_col and site_col in df.columns:
        df[site_col] = _safe_series(df[site_col]).str.strip()
    if city_col and city_col in df.columns:
        df[city_col] = _safe_series(df[city_col]).str.strip()
    if uf_col and uf_col in df.columns:
        df[uf_col] = _safe_series(df[uf_col]).str.strip()

    # preço/m3 numérico
    if ppm3_col and ppm3_col in df.columns:
        df[ppm3_col] = _to_float_series(df[ppm3_col])

    # =========================
    # Opções estáveis
    # =========================
    status_all = sorted(_safe_series(df[status_col]).dropna().unique().tolist()) if status_col else []

    # =========================
    # Filtros globais (topo) — empresa respeita Status (e Município)
    # =========================
    st.markdown("#### Filtros globais")
    r1c1, r1c2, r1c3, r1c4 = st.columns([2.2, 2.2, 2.2, 1.6])

    # 1) Status primeiro
    with r1c3:
        sel_status = st.multiselect("Status", options=status_all, placeholder="Opcional") if status_all else []

    # 2) filtros rápidos
    with r1c4:
        only_found = st.checkbox("Somente FOUND", value=False)
        has_ppm3 = st.checkbox("Somente com preço/m3", value=False)

    # 3) df auxiliar para opções dependentes
    df_opts = df.copy()
    if status_col and sel_status:
        df_opts = df_opts[_safe_series(df_opts[status_col]).isin(sel_status)]
    if only_found and status_col:
        df_opts = df_opts[_safe_series(df_opts[status_col]).str.upper().eq("FOUND")]

    # 4) Município depende de status/only_found
    cidades_all = sorted(_safe_series(df_opts[city_col]).dropna().unique().tolist()) if city_col else []
    with r1c2:
        sel_cidades = st.multiselect("Município", options=cidades_all, placeholder="Opcional") if cidades_all else []

    # 5) Empresa depende de status/only_found e também de município
    df_opts2 = df_opts.copy()
    if city_col and sel_cidades:
        df_opts2 = df_opts2[_safe_series(df_opts2[city_col]).isin(sel_cidades)]
    empresas_all = sorted(_safe_series(df_opts2[nm_col]).dropna().unique().tolist())

    with r1c1:
        sel_empresas = st.multiselect("Empresa", options=empresas_all, placeholder="Selecione uma ou mais")

    r2c1, r2c2, r2c3 = st.columns([2.0, 2.0, 2.0])
    with r2c1:
        busca_produto = st.text_input("Busca produto (nome)", placeholder="ex: painel pinus").strip()

    with r2c2:
        # faixa de preço/m3
        range_ppm3 = None
        if ppm3_col and df[ppm3_col].notna().any():
            vmin = float(df[ppm3_col].min())
            vmax = float(df[ppm3_col].max())
            if vmin == vmax:
                st.caption("Preço/m3 sem variação no dataset.")
            else:
                range_ppm3 = st.slider("Faixa preço/m3", float(vmin), float(vmax), (float(vmin), float(vmax)))

    with r2c3:
        st.caption("")

    st.divider()

    # =========================
    # Aplicar filtros no DF (produtos)
    # =========================
    filtered = df.copy()

    # Status/only_found primeiro (para coerência total)
    if status_col and sel_status:
        filtered = filtered[_safe_series(filtered[status_col]).isin(sel_status)]
    if only_found and status_col:
        filtered = filtered[_safe_series(filtered[status_col]).str.upper().eq("FOUND")]

    if city_col and sel_cidades:
        filtered = filtered[_safe_series(filtered[city_col]).isin(sel_cidades)]

    if sel_empresas:
        filtered = filtered[_safe_series(filtered[nm_col]).isin(sel_empresas)]

    if has_ppm3 and ppm3_col:
        filtered = filtered[filtered[ppm3_col].notna()]

    if range_ppm3 and ppm3_col:
        filtered = filtered[filtered[ppm3_col].between(range_ppm3[0], range_ppm3[1])]

    if busca_produto and prod_name_col and prod_name_col in filtered.columns:
        filtered = filtered[_safe_series(filtered[prod_name_col]).str.contains(busca_produto, case=False, na=False)]

    # =========================
    # KPIs (estilo portfolio)
    # =========================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Empresas (filtradas)", int(filtered[nm_col].nunique()))
    k2.metric("Produtos (filtrados)", int(len(filtered)))
    if ppm3_col and filtered[ppm3_col].notna().any():
        k3.metric("Preço/m3 (mediana)", round(float(filtered[ppm3_col].median()), 2))
        k4.metric("Preço/m3 (mín)", round(float(filtered[ppm3_col].min()), 2))
    else:
        k3.metric("Preço/m3 (mediana)", "-")
        k4.metric("Preço/m3 (mín)", "-")

    st.divider()

    # =========================
    # EMPRESAS: 2 colunas
    # - ESQUERDA: cards (2x2 por viewport) com scroll
    # - DIREITA: detalhe da empresa
    # =========================
    st.markdown("#### Empresas")

    # dataset de empresas (a partir do filtered)
    comp_view = (
        filtered.groupby(nm_col, dropna=False)
        .agg(
            website_url=(site_col, "first") if site_col else (nm_col, "first"),
            municipio=(city_col, "first") if city_col else (nm_col, "first"),
            uf=(uf_col, "first") if uf_col else (nm_col, "first"),
            status=(status_col, "first") if status_col else (nm_col, "first"),
            produtos=(nm_col, "size"),
            ppm3_mediana=(ppm3_col, "median") if ppm3_col else (nm_col, "first"),
            msg=(msg_col, "first") if msg_col else (nm_col, "first"),
        )
        .reset_index()
    )

    # ordenação por volume
    if "produtos" in comp_view.columns:
        comp_view = comp_view.sort_values("produtos", ascending=False)

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.caption("Cards (2x2). Role dentro do container para ver mais empresas.")
        with st.container(height=800, border=False):
            empresas_cards_df = comp_view.copy()

            for start in range(0, len(empresas_cards_df), 4):
                page = empresas_cards_df.iloc[start : start + 4]

                c1, c2 = st.columns(2, gap="large")
                for i, (_, rr) in enumerate(page.iterrows()):
                    target = c1 if i % 2 == 0 else c2
                    with target:
                        empresa = str(rr.get(nm_col, "") or "")
                        mun = str(rr.get("municipio", "") or "")
                        uf = str(rr.get("uf", "") or "")
                        label_loc = f"{mun} - {uf}".strip(" -") if (mun or uf) else ""
                        nome_fmt = f"{empresa} — {label_loc}" if label_loc else empresa

                        url = str(rr.get("website_url", "") or "")
                        resumo = str(rr.get("msg", "") or "")

                        produtos = int(rr.get("produtos", 0) or 0)
                        stt = str(rr.get("status", "") or "")
                        ppm3_med = rr.get("ppm3_mediana", None)
                        ppm3_med = round(float(ppm3_med), 2) if ppm3_col and pd.notna(ppm3_med) else "—"

                        _card_empresa(
                            nome=nome_fmt,
                            url=url,
                            resumo=resumo,
                            kpis={
                                "Produtos": produtos,
                                "Status": stt or "—",
                                "Preço/m3 (med)": ppm3_med,
                            },
                        )

    with right:
        st.caption("Detalhe da empresa")

        empresas_opts = comp_view[nm_col].dropna().unique().tolist()
        empresa_sel = st.selectbox("Escolha uma empresa", options=empresas_opts, index=0) if empresas_opts else None

        if not empresa_sel:
            st.info("Nenhuma empresa disponível com os filtros atuais.")
        else:
            row = comp_view[comp_view[nm_col] == empresa_sel].head(1)
            r = row.iloc[0]

            url = str(r.get("website_url", "") or "")
            msg = str(r.get("msg", "") or "")
            mun = str(r.get("municipio", "") or "")
            uf = str(r.get("uf", "") or "")
            stt = str(r.get("status", "") or "")

            st.markdown(f"##### {empresa_sel}")
            if mun or uf:
                st.caption(f"📍 {mun} - {uf}".strip(" -"))
            if stt:
                st.caption(f"🟦 Status: {stt}")
            if url:
                st.markdown(f"[Abrir site]({url})")

            st.divider()
            st.markdown("**Resumo**")
            st.markdown(_pretty_multiline(msg))

            st.divider()
            cA, cB = st.columns(2)
            cA.metric("Produtos (filtrados)", int(filtered[_safe_series(filtered[nm_col]) == empresa_sel].shape[0]))

            if ppm3_col:
                sub = filtered[_safe_series(filtered[nm_col]) == empresa_sel]
                if sub[ppm3_col].notna().any():
                    cB.metric("Preço/m3 (mediana)", round(float(sub[ppm3_col].median()), 2))
                else:
                    cB.metric("Preço/m3 (mediana)", "-")
            else:
                cB.metric("Preço/m3 (mediana)", "-")

    st.divider()

    # =========================
    # PRODUTOS (detalhe) + preview
    # =========================
    st.markdown("#### Produtos (detalhe)")

    display_df = filtered.copy()

    # Monta URL completa do produto quando possível
    if site_col and prod_url_col and site_col in display_df.columns and prod_url_col in display_df.columns:
        base_urls = _safe_series(display_df[site_col]).str.strip()
        product_urls = _safe_series(display_df[prod_url_col]).str.strip()
        display_df["product_url"] = [
            urljoin(base, path) if path else base
            for base, path in zip(base_urls, product_urls)
        ]

    desired_cols = [
        nm_col,
        site_col,
        city_col,
        "name" if "name" in display_df.columns else prod_name_col,
        dims_raw_col,
        price_col,
        ppm3_col,
        "product_url" if "product_url" in display_df.columns else prod_url_col,
        status_col,
    ]
    table_cols = [c for c in desired_cols if c and c in display_df.columns]

    st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True, height=560)

    # Preview do produto
    if "product_url" in display_df.columns:
        st.markdown("#### Preview do produto")
        preview_options = display_df["product_url"].dropna().unique().tolist()
        if preview_options:
            selected_url = st.selectbox("Selecionar URL para visualizar", preview_options)
            if selected_url:
                st.components.v1.iframe(selected_url, height=520, scrolling=True)
        else:
            st.info("Não há URLs de produto disponíveis para preview.")

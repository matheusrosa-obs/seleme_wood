import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Callable


def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _normalize_yes_no(s: pd.Series) -> pd.Series:
    ss = _safe_str_series(s).str.strip().str.lower()
    return ss.isin(["1", "true", "yes", "sim", "y", "t"])


def _safe_text_contains(df: pd.DataFrame, cols: list[str], query: str) -> pd.Series:
    if not query:
        return pd.Series(True, index=df.index)
    hay = pd.Series("", index=df.index)
    for c in cols:
        if c in df.columns:
            hay = hay + " " + _safe_str_series(df[c])
    return hay.str.contains(query, case=False, na=False)


def _pretty_multiline(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "—"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return "\n".join([f"- {ln.lstrip('-• ').strip()}" for ln in lines])
    return t


def _split_empresa_municipio(s: pd.Series) -> pd.DataFrame:
    ss = _safe_str_series(s).str.strip()
    parts = ss.str.rsplit(" - ", n=1, expand=True)
    if parts.shape[1] == 1:
        empresa_nome = parts[0].fillna("").str.strip()
        municipio = ""
    else:
        empresa_nome = parts[0].fillna("").str.strip()
        municipio = parts[1].fillna("").str.strip()
    return pd.DataFrame({"empresa_nome": empresa_nome, "municipio": municipio})


def _inject_cards_css_once() -> None:
    st.markdown(
        """
        <style>
        .cards-scroll-800 {
            max-height: 800px;
            overflow-y: auto;
            padding-right: 8px;
            padding-top: 2px;
        }
        .cards-scroll-800::-webkit-scrollbar { width: 9px; }
        .cards-scroll-800::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.18);
            border-radius: 999px;
        }
        .cards-scroll-800::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.06);
            border-radius: 999px;
        }

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

    pills = []
    for label, val in kpis.items():
        pills.append(f"<span class='pill'>{label}: <b>{val}</b></span>")
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


def render_tab_portfolio(
    load_data_fn: Callable[[str], pd.DataFrame],
    placeholder_df: pd.DataFrame | None,
    companies_path: str,
    items_path: str,
) -> None:
    st.markdown("### Portfólio")
    st.caption("Filtros globais no topo e duas tabelas: Empresas (cards) e Produtos (items).")

    # =========================
    # Carregar CSVs
    # =========================
    try:
        df_companies = load_data_fn(companies_path)
        df_items = load_data_fn(items_path)
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        st.stop()

    if df_companies.empty:
        st.warning("companies.csv está vazio.")
        return

    if df_items.empty:
        st.warning("items.csv está vazio. A tabela de produtos ficará vazia.")

    if "nome_empresa" not in df_companies.columns:
        st.error("companies.csv não contém a coluna obrigatória: 'nome_empresa'")
        return
    if "nome_empresa" not in df_items.columns:
        st.error("items.csv não contém a coluna obrigatória: 'nome_empresa'")
        return

    # =========================
    # Normalizações + split empresa/município
    # =========================
    df_companies = df_companies.copy()
    df_items = df_items.copy()

    df_companies["nome_empresa"] = _safe_str_series(df_companies["nome_empresa"]).str.strip()
    df_items["nome_empresa"] = _safe_str_series(df_items["nome_empresa"]).str.strip()

    comp_split = _split_empresa_municipio(df_companies["nome_empresa"])
    df_companies["empresa_nome"] = comp_split["empresa_nome"]
    df_companies["municipio"] = comp_split["municipio"]

    it_split = _split_empresa_municipio(df_items["nome_empresa"])
    df_items["empresa_nome"] = it_split["empresa_nome"]
    df_items["municipio"] = it_split["municipio"]

    for col in ["qtde_itens_total", "qtde_itens_madeira", "qtde_itens_menciona_pinus"]:
        if col in df_companies.columns:
            df_companies[col] = _to_int_series(df_companies[col])

    # =========================
    # 🔧 UF: robusto (tenta achar coluna existente)
    # =========================
    uf_col = None
    for cand in ["nm_uf"]:
        if cand in df_companies.columns:
            uf_col = cand
            break

    if uf_col is not None:
        df_companies["_uf_"] = _safe_str_series(df_companies[uf_col]).str.strip().str.upper()
    else:
        df_companies["_uf_"] = ""  # mantém sempre a coluna para não quebrar UI

    uf_all = sorted([u for u in df_companies["_uf_"].dropna().unique() if u in ["PARANÁ", "RIO GRANDE DO SUL", "SANTA CATARINA"]])

    # =========================
    # Opções para filtros
    # =========================
    empresas_all = sorted(df_companies["nome_empresa"].dropna().unique().tolist())

    municipios_all = (
        sorted(df_companies["municipio"].dropna().astype(str).unique().tolist())
        if "municipio" in df_companies.columns
        else []
    )

    cat_norm_all = (
        sorted(df_items["categoria_produto_norm"].dropna().unique().tolist())
        if "categoria_produto_norm" in df_items.columns
        else []
    )
    cat_grupo_all = (
        sorted(df_items["categoria_produto_grupo"].dropna().unique().tolist())
        if "categoria_produto_grupo" in df_items.columns
        else []
    )
    mat_norm_all = (
        sorted(df_items["material_norm"].dropna().unique().tolist())
        if "material_norm" in df_items.columns
        else []
    )
    mat_grupo_all = (
        sorted(df_items["material_grupo"].dropna().unique().tolist())
        if "material_grupo" in df_items.columns
        else []
    )

    total_min, total_max = 0, 0
    if "qtde_itens_total" in df_companies.columns and len(df_companies) > 0:
        total_min = int(df_companies["qtde_itens_total"].min())
        total_max = int(df_companies["qtde_itens_total"].max())

    # =========================
    # Filtros globais (topo)
    # ✅ Ordem UX: Localização -> Empresa -> Produto -> Flags -> Quantidade
    # =========================
    st.markdown("#### Filtros globais")

    r1c1, r1c2, r1c3, r1c4 = st.columns([1.4, 2.2, 2.2, 1.6])

    with r1c1:
        # ✅ UF em posição "natural" (localização vem primeiro)
        sel_uf = st.multiselect("UF", options=uf_all, placeholder="Opcional") if uf_all else []
        sel_municipios = (
            st.multiselect("Município", options=municipios_all, placeholder="Opcional")
            if municipios_all
            else []
        )

    with r1c2:
        sel_empresas = st.multiselect(
            "Empresa",
            options=empresas_all,
            placeholder="Selecione uma ou mais empresas",
        )

    with r1c3:
        sel_cat_norm = st.multiselect("Categoria", options=cat_norm_all, placeholder="Opcional") if cat_norm_all else []
        sel_cat_grupo = st.multiselect("Categoria (grupo)", options=cat_grupo_all, placeholder="Opcional") if cat_grupo_all else []

    with r1c4:
        sel_mat_norm = st.multiselect("Material", options=mat_norm_all, placeholder="Opcional") if mat_norm_all else []
        sel_mat_grupo = st.multiselect("Material (grupo)", options=mat_grupo_all, placeholder="Opcional") if mat_grupo_all else []
        only_pinus = st.checkbox("Somente com indício de pinus", value=True)
        only_madeira = st.checkbox("Somente trabalha com madeira", value=False)

    r2c1, _, _ = st.columns([2.0, 2.0, 2.0])
    with r2c1:
        if total_max > 0:
            range_total = st.slider("Faixa Itens (Total)", total_min, total_max, (total_min, total_max))
        else:
            range_total = None

    st.divider()

    # =========================
    # Aplicar filtros em COMPANIES
    # =========================
    comp = df_companies.copy()

    if sel_uf:
        comp = comp[comp["_uf_"].isin(sel_uf)]

    if sel_empresas:
        comp = comp[comp["nome_empresa"].isin(sel_empresas)]

    if sel_municipios and "municipio" in comp.columns:
        comp = comp[comp["municipio"].astype(str).isin(sel_municipios)]

    if range_total and "qtde_itens_total" in comp.columns:
        comp = comp[comp["qtde_itens_total"].between(range_total[0], range_total[1])]

    if only_pinus:
        mask = pd.Series(False, index=comp.index)
        if "trabalha_com_painel_pinus" in comp.columns:
            mask = mask | _normalize_yes_no(comp["trabalha_com_painel_pinus"])
        if "flag_painel_pinus" in comp.columns:
            mask = mask | (_to_int_series(comp["flag_painel_pinus"]) == 1)
        if "qtde_itens_menciona_pinus" in comp.columns:
            mask = mask | (comp["qtde_itens_menciona_pinus"] > 0)
        comp = comp[mask]

    if only_madeira and "trabalha_com_madeira" in comp.columns:
        comp = comp[_normalize_yes_no(comp["trabalha_com_madeira"])]

    # =========================
    # Aplicar filtros em ITEMS
    # =========================
    it = df_items.copy()

    companies_were_filtered = any(
        [
            bool(sel_uf),
            bool(sel_empresas),
            bool(sel_municipios),
            bool(range_total),
            bool(only_pinus),
            bool(only_madeira),
        ]
    )

    if companies_were_filtered:
        empresas_filtradas = comp["nome_empresa"].dropna().unique().tolist()
        it = it[it["nome_empresa"].isin(empresas_filtradas)]

    if sel_cat_norm and "categoria_produto_norm" in it.columns:
        it = it[it["categoria_produto_norm"].isin(sel_cat_norm)]
    if sel_cat_grupo and "categoria_produto_grupo" in it.columns:
        it = it[it["categoria_produto_grupo"].isin(sel_cat_grupo)]
    if sel_mat_norm and "material_norm" in it.columns:
        it = it[it["material_norm"].isin(sel_mat_norm)]
    if sel_mat_grupo and "material_grupo" in it.columns:
        it = it[it["material_grupo"].isin(sel_mat_grupo)]

    # =========================
    # KPIs
    # =========================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Empresas (filtradas)", int(comp["nome_empresa"].nunique()) if "nome_empresa" in comp.columns else 0)
    k2.metric("Produtos (filtrados)", int(len(it)))
    k3.metric("Categorias", int(it["categoria_produto"].nunique()) if "categoria_produto" in it.columns else "-")
    k4.metric("Materiais (grupo)", int(it["material_grupo"].nunique()) if "material_grupo" in it.columns else "-")

    st.divider()

    # =========================
    # EMPRESAS (cards)
    # =========================
    st.markdown("#### Empresas")

    _inject_cards_css_once()

    def _first_non_empty(series: pd.Series) -> str:
        s = _safe_str_series(series).str.strip()
        s = s[s != ""]
        return s.iloc[0] if len(s) else ""

    agg_dict: dict = {}
    for c in ["empresa_nome", "municipio", "url_site", "resumo_materiais", "nome_empresa"]:
        if c in comp.columns:
            agg_dict[c] = _first_non_empty

    for c in ["qtde_itens_total", "qtde_itens_madeira", "qtde_itens_menciona_pinus"]:
        if c in comp.columns:
            agg_dict[c] = "max"

    if "nome_empresa" in comp.columns:
        comp_view = comp.groupby("nome_empresa", dropna=False, as_index=False).agg(agg_dict)
    else:
        comp_view = comp.copy()

    if "qtde_itens_total" in comp_view.columns:
        comp_view = comp_view.sort_values("qtde_itens_total", ascending=False)

    with st.container(height=550, border=False):
        empresas_cards_df = comp_view.copy()

        for start in range(0, len(empresas_cards_df), 4):
            page = empresas_cards_df.iloc[start : start + 4]

            c1, c2 = st.columns(2, gap="large")
            for i, (_, rr) in enumerate(page.iterrows()):
                target = c1 if i % 2 == 0 else c2
                with target:
                    nome = str(rr.get("empresa_nome", "") or rr.get("nome_empresa", "") or "")
                    municipio = str(rr.get("municipio", "") or "")
                    uf_val = str(rr.get("_uf_", "") or "")  # pode estar vazio aqui
                    nome_fmt = f"{nome} — {municipio}" if municipio else nome
                    if uf_val:
                        nome_fmt = f"{nome_fmt} / {uf_val}"

                    _card_empresa(
                        nome=nome_fmt,
                        url=str(rr.get("url_site", "") or ""),
                        resumo=str(rr.get("resumo_materiais", "") or ""),
                        kpis={
                            "Total": int(rr.get("qtde_itens_total", 0) or 0),
                            "Madeira": int(rr.get("qtde_itens_madeira", 0) or 0),
                            "Pinus": int(rr.get("qtde_itens_menciona_pinus", 0) or 0),
                        },
                    )

    st.divider()

    # =========================
    # PRODUTOS (st.dataframe)
    # =========================
    st.markdown("#### Produtos")

    cols_items_hard = [
        "empresa_nome",
        "municipio",
        "nome_produto",
        "categoria_produto",
        "categoria_produto_grupo",
        "material",
        "material_grupo",
    ]
    cols_items_hard = [c for c in cols_items_hard if c in it.columns]

    if "empresa_nome" in it.columns and "categoria_produto_grupo" in it.columns:
        it_view = it.sort_values(["empresa_nome", "categoria_produto_grupo"], ascending=True)
    else:
        it_view = it.copy()

    st.dataframe(
        it_view[cols_items_hard].head(5000),
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    # =========================
    # MAPA (placeholder) — opcional
    # =========================
    if placeholder_df is not None and not placeholder_df.empty:
        st.divider()
        st.markdown("#### Mapa (placeholder)")

        if all(c in placeholder_df.columns for c in ["nu_latitude", "nu_longitude"]):
            fig_p = px.scatter_mapbox(
                placeholder_df,
                lat="nu_latitude",
                lon="nu_longitude",
                size="Quantidade" if "Quantidade" in placeholder_df.columns else None,
                color="UF" if "UF" in placeholder_df.columns else None,
                hover_name="Município" if "Município" in placeholder_df.columns else None,
                center={"lat": -27.0, "lon": -50.0},
                zoom=5,
                size_max=30,
            )
            fig_p.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("placeholder_df não possui nu_latitude/nu_longitude para exibir o mapa.")

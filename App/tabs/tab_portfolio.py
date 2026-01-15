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
    """
    Melhora apresentação do resumo: preserva quebras e tenta transformar em bullets quando fizer sentido.
    """
    t = (text or "").strip()
    if not t:
        return "—"

    # Se já tiver bullets ou quebras, mantém bem apresentado
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return "\n".join([f"- {ln.lstrip('-• ').strip()}" for ln in lines])

    # Se for uma frase longa, devolve como parágrafo
    return t


def _card_empresa(nome: str, url: str, resumo: str, kpis: dict) -> None:
    """
    Card simples e agradável para mostrar resumo_materiais.
    """
    resumo_fmt = _pretty_multiline(resumo)

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

    url_html = f"<a href='{url}' target='_blank' style='text-decoration: underline; opacity:0.95;'>{url}</a>" if url else "<span style='opacity:0.75'>—</span>"

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
    placeholder_df: pd.DataFrame,
    companies_path: str,
    items_path: str,
) -> None:
    st.markdown("### Portfólio")
    st.caption("Filtros globais no topo e duas tabelas: Empresas (companies) e Produtos (items).")

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

    # Normalizações
    df_companies["nome_empresa"] = _safe_str_series(df_companies["nome_empresa"]).str.strip()
    df_items["nome_empresa"] = _safe_str_series(df_items["nome_empresa"]).str.strip()

    for col in ["qtde_itens_total", "qtde_itens_madeira", "qtde_itens_menciona_pinus"]:
        if col in df_companies.columns:
            df_companies[col] = _to_int_series(df_companies[col])

    # =========================
    # Opções para filtros
    # =========================
    empresas_all = sorted(df_companies["nome_empresa"].dropna().unique().tolist())

    status_all = (
        sorted(df_companies["status_execucao"].dropna().astype(str).unique().tolist())
        if "status_execucao" in df_companies.columns
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
    # =========================
    st.markdown("#### Filtros globais")

    r1c1, r1c2, r1c3, r1c4 = st.columns([2.2, 2.2, 2.2, 1.6])
    with r1c1:
        sel_empresas = st.multiselect("Empresa", options=empresas_all, placeholder="Selecione uma ou mais empresas")
        busca_empresa = st.text_input("Busca por empresa", placeholder="Digite parte do nome...").strip()

    with r1c2:
        sel_cat_norm = st.multiselect("Categoria (norm)", options=cat_norm_all, placeholder="Opcional") if cat_norm_all else []
        sel_cat_grupo = st.multiselect("Categoria (grupo)", options=cat_grupo_all, placeholder="Opcional") if cat_grupo_all else []

    with r1c3:
        sel_mat_norm = st.multiselect("Material (norm)", options=mat_norm_all, placeholder="Opcional") if mat_norm_all else []
        sel_mat_grupo = st.multiselect("Material (grupo)", options=mat_grupo_all, placeholder="Opcional") if mat_grupo_all else []

    with r1c4:
        sel_status = st.multiselect("Status", options=status_all, placeholder="Opcional") if status_all else []
        only_pinus = st.checkbox("Somente com indício de pinus", value=True)
        only_madeira = st.checkbox("Somente trabalha com madeira", value=False)

    r2c1, r2c2, r2c3 = st.columns([2.0, 2.0, 2.0])
    with r2c1:
        if total_max > 0:
            range_total = st.slider("Faixa Itens (Total)", total_min, total_max, (total_min, total_max))
        else:
            range_total = None

    with r2c2:
        only_prod_menciona_pinus = st.checkbox("Produtos: somente menciona pinus", value=False)
        only_prod_rel_madeira = st.checkbox("Produtos: somente relacionados à madeira", value=False)

    with r2c3:
        busca_produto = st.text_input("Busca produto (nome/categoria/material)", placeholder="ex: painel pinus 18mm").strip()

    st.divider()

    # =========================
    # Aplicar filtros em COMPANIES
    # =========================
    comp = df_companies.copy()

    if sel_empresas:
        comp = comp[comp["nome_empresa"].isin(sel_empresas)]

    if busca_empresa:
        comp = comp[_safe_str_series(comp["nome_empresa"]).str.contains(busca_empresa, case=False, na=False)]

    if sel_status and "status_execucao" in comp.columns:
        comp = comp[comp["status_execucao"].astype(str).isin(sel_status)]

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
            bool(sel_empresas),
            bool(busca_empresa),
            bool(sel_status),
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

    if only_prod_menciona_pinus and "menciona_pinus" in it.columns:
        if pd.api.types.is_numeric_dtype(it["menciona_pinus"]):
            it = it[it["menciona_pinus"].fillna(0).astype(int) == 1]
        else:
            it = it[it[_normalize_yes_no(it["menciona_pinus"])]]

    if only_prod_rel_madeira and "relacionado_madeira" in it.columns:
        if pd.api.types.is_numeric_dtype(it["relacionado_madeira"]):
            it = it[it["relacionado_madeira"].fillna(0).astype(int) == 1]
        else:
            it = it[it[_normalize_yes_no(it["relacionado_madeira"])]]

    if busca_produto:
        mask_txt = _safe_text_contains(
            it,
            cols=[
                "nome_produto",
                "categoria_produto",
                "categoria_produto_norm",
                "material",
                "material_norm",
                "material_grupo",
            ],
            query=busca_produto,
        )
        it = it[mask_txt]

    # =========================
    # KPIs
    # =========================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Empresas (filtradas)", int(comp["nome_empresa"].nunique()) if "nome_empresa" in comp.columns else 0)
    k2.metric("Produtos (filtrados)", int(len(it)))
    k3.metric("Categorias (grupo)", int(it["categoria_produto_grupo"].nunique()) if "categoria_produto_grupo" in it.columns else "-")
    k4.metric("Materiais (grupo)", int(it["material_grupo"].nunique()) if "material_grupo" in it.columns else "-")

    st.divider()

    # =========================
    # EMPRESAS: UX melhor
    # - tabela enxuta (sem resumo_materiais)
    # - painel de detalhe com resumo_materiais bem formatado
    # =========================
    st.markdown("#### Empresas")

    # tabela enxuta (hardcode)
    cols_companies_table = [
        "nome_empresa",
        "url_site",
        "qtde_itens_total",
        "qtde_itens_madeira",
        "qtde_itens_menciona_pinus",
    ]
    cols_companies_table = [c for c in cols_companies_table if c in comp.columns]

    # Ordena por total
    comp_view = comp.sort_values("qtde_itens_total", ascending=False) if "qtde_itens_total" in comp.columns else comp.copy()

    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.caption("Tabela rápida (métricas). Selecione uma empresa no painel ao lado para ver o resumo.")
        st.dataframe(
            comp_view[cols_companies_table].head(1500),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    with right:
        st.caption("Detalhe da empresa (resumo_materiais)")

        empresas_opts = comp_view["nome_empresa"].dropna().unique().tolist()
        default_opt = empresas_opts[0] if empresas_opts else None

        empresa_sel = st.selectbox(
            "Escolha uma empresa",
            options=empresas_opts,
            index=0 if default_opt else 0,
        ) if empresas_opts else None

        if not empresa_sel:
            st.info("Nenhuma empresa disponível com os filtros atuais.")
        else:
            row = comp_view[comp_view["nome_empresa"] == empresa_sel].head(1)
            if row.empty:
                st.info("Empresa não encontrada (cheque filtros).")
            else:
                r = row.iloc[0]
                url = str(r.get("url_site", "") or "")
                resumo = str(r.get("resumo_materiais", "") or "")

                # cabeçalho
                st.markdown(f"##### {empresa_sel}")
                if url:
                    st.markdown(f"[Abrir site]({url})")
                else:
                    st.caption("Sem url_site.")

                # “tags”/pills com contadores
                cA, cB, cC = st.columns(3)
                cA.metric("Itens total", int(r.get("qtde_itens_total", 0)))
                cB.metric("Itens madeira", int(r.get("qtde_itens_madeira", 0)))
                cC.metric("Menciona pinus", int(r.get("qtde_itens_menciona_pinus", 0)))

                st.divider()
                st.markdown("**Resumo de materiais**")

                # texto bonito (bullets se multilinha)
                st.markdown(_pretty_multiline(resumo))

                # Alternativa opcional: cards para todas (boa pra poucos resultados)
                st.divider()
                show_cards = st.checkbox("Mostrar também como cards (lista)", value=False)
                if show_cards:
                    st.caption("Cards ficam bons quando o filtro retorna poucas empresas (ex.: < 50).")
                    # cria uma grade simples de cards (2 colunas)
                    cards_df = comp_view.head(50).copy()
                    c1, c2 = st.columns(2)
                    for i, (_, rr) in enumerate(cards_df.iterrows()):
                        target = c1 if i % 2 == 0 else c2
                        with target:
                            _card_empresa(
                                nome=str(rr.get("nome_empresa", "")),
                                url=str(rr.get("url_site", "")),
                                resumo=str(rr.get("resumo_materiais", "")),
                                kpis={
                                    "Total": int(rr.get("qtde_itens_total", 0)),
                                    "Madeira": int(rr.get("qtde_itens_madeira", 0)),
                                    "Pinus": int(rr.get("qtde_itens_menciona_pinus", 0)),
                                },
                            )

    st.divider()

    # =========================
    # PRODUTOS (st.dataframe)
    # =========================
    st.markdown("#### Produtos")

    cols_items_hard = [
        "nome_empresa",
        "url_site",
        "nome_produto",
        "categoria_produto",
        "categoria_produto_norm",
        "categoria_produto_grupo",
        "material",
        "material_norm",
        "material_grupo",
        "relacionado_madeira",
        "menciona_pinus",
        "fonte_indicio",
        "data_coleta",
    ]
    cols_items_hard = [c for c in cols_items_hard if c in it.columns]

    if "nome_empresa" in it.columns and "categoria_produto_grupo" in it.columns:
        it_view = it.sort_values(["nome_empresa", "categoria_produto_grupo"], ascending=True)
    else:
        it_view = it.copy()

    st.dataframe(
        it_view[cols_items_hard].head(5000),
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    # =========================
    # Mapa placeholder (de volta)
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
                height=500,
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("placeholder_df não possui nu_latitude/nu_longitude para exibir o mapa.")

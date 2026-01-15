import streamlit as st
import plotly.express as px
import pandas as pd


def render_tab_ecom(placeholder_df: pd.DataFrame) -> None:
    st.markdown(
        "Breve explicação: nesta aba colocaremos os e-commerces que atuam como consumidores ou revendedores. (Placeholder)"
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Tabela (placeholder)**")
        st.dataframe(
            placeholder_df[["Município", "UF", "Quantidade"]],
            height=300,
            hide_index=True,
        )

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

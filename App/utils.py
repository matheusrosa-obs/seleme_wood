import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import matplotlib.colors as mcolors
from typing import Dict, Tuple, Optional, List
import requests
import traceback
#from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import requests
import streamlit as st


class CalculadoraDistanciasMultiRef:
    """
    Calcula distâncias em linha reta de várias referências (portos)
    para todas as empresas.

    Cada referência é um dict com:
        {
            "id": str,
            "nome": str,
            "sg_uf": str,
            "latitude": float,
            "longitude": float
        }
    """

    def __init__(self, df_empresas: pd.DataFrame, referencias: List[Dict]):
        print("\n" + "="*60)
        print("🧮 [LOG] INICIALIZANDO CALCULADORA DE DISTÂNCIAS (MULTI-REF)")
        print("="*60)

        self.referencias = referencias
        # Remove linhas sem latitude/longitude
        self.df = df_empresas.dropna(subset=["latitude", "longitude"]).copy()

        registros_removidos = len(df_empresas) - len(self.df)

        print("📊 [LOG] Dados válidos:")
        print(f"   ✅ [LOG] Pontos com coordenadas: {len(self.df):,}")
        print(f"   ❌ [LOG] Pontos removidos: {registros_removidos:,}")
        print("="*60 + "\n")

    def calcular_todas_distancias(self) -> pd.DataFrame:
        """
        Para cada empresa, calcula a distância geodésica até cada referência
        e guarda também a distância mínima e a referência mais próxima.

        Saída:
            df_dist com colunas:
                - dist_<id_ref>_km   (uma coluna por porto)
                - distancia_km       (distância mínima)
                - ref_mais_proxima_id
                - ref_mais_proxima_nome
                - ref_mais_proxima_uf
        """
        print("🔄 [LOG] Calculando distâncias para todas as referências...")
        distancias = []
        total = len(self.df)

        for i, (_, row) in enumerate(self.df.iterrows(), start=1):
            coord_empresa = (row["latitude"], row["longitude"])

            info_base = {
                "nu_cnpj": row["nu_cnpj"],
                "nm_nome_fantasia": row["nm_nome_fantasia"],
                "nm_razao_social": row["nm_razao_social"],
                "nm_mun": row["nm_mun"],
                "sg_uf": row["sg_uf"],
                "cd_mun_ibge": row.get("cd_mun_ibge", ""),
                "nm_porte_obs": row.get("nm_porte_obs", ""),
                "cd_cnae_fiscal_principal": row.get("cd_cnae_fiscal_principal", ""),
                "nm_cnae_fiscal_principal": row.get("nm_cnae_fiscal_principal", ""),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }

            dist_min = None
            ref_min = None

            # Distância até cada porto
            for ref in self.referencias:
                coord_ref = (ref["latitude"], ref["longitude"])
                d_km = geodesic(coord_ref, coord_empresa).km

                col_dist = f"dist_{ref['id']}_km"
                info_base[col_dist] = d_km

                if (dist_min is None) or (d_km < dist_min):
                    dist_min = d_km
                    ref_min = ref

            # Distância mínima e referência mais próxima
            info_base["distancia_km"] = dist_min
            info_base["ref_mais_proxima_id"] = ref_min["id"]
            info_base["ref_mais_proxima_nome"] = ref_min["nome"]
            info_base["ref_mais_proxima_uf"] = ref_min["sg_uf"]

            distancias.append(info_base)

            if i % 20 == 0:
                print(f"   ⏳ [LOG] Progresso: {i}/{total} ({i/total*100:.1f}%)")

        df_dist = pd.DataFrame(distancias).sort_values("distancia_km")

        print("\n✅ [LOG] Cálculo concluído!")
        print(f"   • [LOG] Total de distâncias calculadas: {len(df_dist):,}")
        print(f"   • [LOG] Distância mínima: {df_dist['distancia_km'].min():.2f} km")
        print(f"   • [LOG] Distância máxima: {df_dist['distancia_km'].max():.2f} km")

        return df_dist

    def extrair_kpis_completos(self, df_distancias: pd.DataFrame) -> Dict:
        """
        KPIs usando a distância mínima até qualquer porto.
        """
        print("\n📊 [LOG] Extraindo KPIs (multi-ref)...")

        if df_distancias.empty:
            print("⚠️  [LOG] Nenhuma empresa disponível para KPIs.")
            return {"erro": "Sem dados de empresas"}

        def safe_get(row, col, default="(Não informado)"):
            val = row.get(col, None)
            if pd.isna(val) or val == "":
                return default
            return str(val)

        row_proximo = df_distancias.iloc[0]
        row_distante = df_distancias.iloc[-1]

        kpis = {
            "mais_proximo": {
                "cnpj": safe_get(row_proximo, "nu_cnpj"),
                "nome_fantasia": safe_get(row_proximo, "nm_nome_fantasia"),
                "razao_social": safe_get(row_proximo, "nm_razao_social"),
                "municipio": safe_get(row_proximo, "nm_mun"),
                "uf": safe_get(row_proximo, "sg_uf", "BR"),
                "distancia_km": row_proximo["distancia_km"],
                "ref_mais_proxima_nome": safe_get(row_proximo, "ref_mais_proxima_nome"),
                "ref_mais_proxima_uf": safe_get(row_proximo, "ref_mais_proxima_uf"),
            },
            "mais_distante": {
                "cnpj": safe_get(row_distante, "nu_cnpj"),
                "nome_fantasia": safe_get(row_distante, "nm_nome_fantasia"),
                "razao_social": safe_get(row_distante, "nm_razao_social"),
                "municipio": safe_get(row_distante, "nm_mun"),
                "uf": safe_get(row_distante, "sg_uf", "BR"),
                "distancia_km": row_distante["distancia_km"],
                "ref_mais_proxima_nome": safe_get(row_distante, "ref_mais_proxima_nome"),
                "ref_mais_proxima_uf": safe_get(row_distante, "ref_mais_proxima_uf"),
            },
            "estatisticas": {
                "media_km": df_distancias["distancia_km"].mean(),
                "mediana_km": df_distancias["distancia_km"].median(),
                "desvio_padrao_km": df_distancias["distancia_km"].std(),
                "p25_km": df_distancias["distancia_km"].quantile(0.25),
                "p75_km": df_distancias["distancia_km"].quantile(0.75),
                "p90_km": df_distancias["distancia_km"].quantile(0.90),
            },
            "distribuicao": {
                "ate_50km": (df_distancias["distancia_km"] <= 50).sum(),
                "de_50_a_100km": (
                    (df_distancias["distancia_km"] > 50)
                    & (df_distancias["distancia_km"] <= 100)
                ).sum(),
                "de_100_a_200km": (
                    (df_distancias["distancia_km"] > 100)
                    & (df_distancias["distancia_km"] <= 200)
                ).sum(),
                "de_200_a_500km": (
                    (df_distancias["distancia_km"] > 200)
                    & (df_distancias["distancia_km"] <= 500)
                ).sum(),
                "acima_500km": (df_distancias["distancia_km"] > 500).sum(),
            },
            "totais": {
                "total_pontos": len(df_distancias),
            },
        }

        print("✅ [LOG] KPIs extraídos com sucesso")
        print(
            f"   • [LOG] Mais próximo: {kpis['mais_proximo']['nome_fantasia']} "
            f"→ {kpis['mais_proximo']['ref_mais_proxima_nome']} "
            f"({kpis['mais_proximo']['distancia_km']:.2f} km)"
        )

        return kpis


class AnalisadorClusters:
    """
    Identifica agrupamentos geográficos de empresas
    usando a distância mínima (distancia_km) como feature adicional.
    """

    def __init__(self, df_distancias: pd.DataFrame):
        self.df = df_distancias.copy()
        print("\n" + "="*60)
        print("🔍 [LOG] INICIALIZANDO ANÁLISE DE CLUSTERS")
        print("="*60)
        print(f"   [LOG] Pontos para análise: {len(self.df):,}")

    def clustering_dbscan(self, raio_km: float = 100, min_pontos: int = 2) -> pd.DataFrame:
        """
        Clustering baseado em densidade (DBSCAN) com métrica haversine.
        """
        print(f"\n🎯 [LOG] Executando DBSCAN...")
        print(f"   [LOG] Parâmetros: Raio={raio_km} km, Min pontos={min_pontos}")

        coords = self.df[["latitude", "longitude"]].values

        kms_per_radian = 6371.0088
        epsilon = raio_km / kms_per_radian

        db = DBSCAN(eps=epsilon, min_samples=min_pontos, metric="haversine")
        self.df["cluster_dbscan"] = db.fit_predict(np.radians(coords))

        n_clusters = len(set(self.df["cluster_dbscan"])) - (
            1 if -1 in self.df["cluster_dbscan"] else 0
        )
        n_noise = (self.df["cluster_dbscan"] == -1).sum()

        print("\n📊 [LOG] Resultados DBSCAN:")
        print(f"   ✅ [LOG] Clusters identificados: {n_clusters}")
        print(f"   ⚪ [LOG] Pontos isolados: {n_noise}")

        if n_clusters > 0:
            for cluster_id in sorted(self.df["cluster_dbscan"].unique()):
                if cluster_id == -1:
                    continue

                df_cluster = self.df[self.df["cluster_dbscan"] == cluster_id]
                municipios = df_cluster["nm_mun"].value_counts()

                print(f"\n   🔵 [LOG] Cluster {cluster_id}:")
                print(f"      • [LOG] Empresas: {len(df_cluster)}")
                print(
                    f"      • [LOG] Principal município: "
                    f"{municipios.index[0]} ({municipios.iloc[0]} empresas)"
                )
                print(
                    f"      • [LOG] Distância média (mínima até porto): "
                    f"{df_cluster['distancia_km'].mean():.1f} km"
                )

        return self.df

    def clustering_kmeans(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Clustering K-Means usando latitude, longitude e distancia_km.
        """
        print(f"\n🎯 [LOG] Executando K-Means com {n_clusters} clusters...")

        features = self.df[["latitude", "longitude", "distancia_km"]].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["cluster_kmeans"] = kmeans.fit_predict(features_scaled)

        print("\n📊 [LOG] Resultados K-Means:")
        for cluster_id in range(n_clusters):
            df_cluster = self.df[self.df["cluster_kmeans"] == cluster_id]
            print(
                f"   🔵 [LOG] Cluster {cluster_id}: {len(df_cluster)} empresas, "
                f"dist média {df_cluster['distancia_km'].mean():.1f} km"
            )

        return self.df

    def analise_completa(self, raio_dbscan: float = 100) -> pd.DataFrame:
        """
        Executa DBSCAN + KMeans com número de clusters adaptativo.
        """
        print("\n" + "="*60)
        print("🔬 [LOG] ANÁLISE COMPLETA DE CLUSTERS")
        print("="*60)

        self.clustering_dbscan(raio_km=raio_dbscan)
        n_ideal = min(5, max(3, len(self.df) // 20))
        self.clustering_kmeans(n_clusters=n_ideal)

        print("\n" + "="*60)
        print("✅ [LOG] ANÁLISE DE CLUSTERS CONCLUÍDA")
        print("="*60 + "\n")

        return self.df


def obter_rota_ors(
    coord_origem: Tuple[float, float],
    coord_destino: Tuple[float, float],
    api_key: str,
    profile: str = "driving-hgv",
):
    """
    Usa OpenRouteService para obter rota rodoviária entre dois pontos.

    Espera:
        coord_origem/destino: (lat, lon)

    ORS espera:
        coordinates: [[lon, lat], [lon, lat], ...]

    Retorna:
        dict com distance_km, duration_min, geometry (lista [lat, lon])
    """
    print("\n[ORS] Iniciando chamada à OpenRouteService...")
    print(f"[ORS] Origem (lat, lon): {coord_origem}, Destino (lat, lon): {coord_destino}")

    if not api_key or isinstance(api_key, float):
        print("[ORS] ❌ API key não fornecida ou inválida. Não será usada rota rodoviária.")
        return None

    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"

    # Converte (lat, lon) -> [lon, lat] para a ORS
    origem_lonlat = [float(coord_origem[1]), float(coord_origem[0])]
    destino_lonlat = [float(coord_destino[1]), float(coord_destino[0])]

    body = {
        "coordinates": [
            origem_lonlat,
            destino_lonlat,
        ]
    }

    print(f"[ORS] JSON enviado para {url}: {body}")

    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        print(f"[ORS] Status code: {resp.status_code}")

        if resp.status_code != 200:
            print("[ORS] Corpo da resposta (erro/troubleshooting):")
            print(resp.text[:800])

        resp.raise_for_status()
        data = resp.json()

        if "features" not in data or not data["features"]:
            print("[ORS] ❌ Resposta sem 'features'.")
            return None

        feat = data["features"][0]
        summary = feat["properties"]["summary"]
        distance_m = summary.get("distance", 0.0)
        duration_s = summary.get("duration", 0.0)

        coords_line = feat["geometry"]["coordinates"]  # [[lon, lat], ...]
        geometry_latlon = [[c[1], c[0]] for c in coords_line]

        print(
            f"[ORS] ✅ Rota obtida: {distance_m/1000:.2f} km, "
            f"{duration_s/60:.1f} min, {len(geometry_latlon)} pontos na geometria."
        )

        return {
            "distance_km": distance_m / 1000.0,
            "duration_min": duration_s / 60.0,
            "geometry": geometry_latlon,
        }

    except Exception as e:
        print(f"[ORS] ❌ Erro ao chamar ORS: {e}")
        return None


def criar_mapa_distancias_portos(
    df_distancias: pd.DataFrame,
    referencias: List[Dict],
    kpis: Dict,
    nm_fantasia_selecionado: Optional[str] = None,
    ors_api_key: Optional[str] = None,
    porto_id_selecionado: Optional[str] = None,
    rotas_precomputadas: Optional[List[Dict]] = None,
) -> folium.Map:
    """
    Mapa Folium com:
      - vários portos como referência
      - empresas coloridas pela distância mínima até qualquer porto
      - empresa selecionada destacada
      - rotas:
          * se porto_id_selecionado = None → rotas para todos os portos
            e a mais curta em verde escuro
          * se porto_id_selecionado = X → rota apenas para o porto X (verde escuro)

    Mapa focado em Santa Catarina (zoom maior, ruas/cidades mais visíveis).

    Se rotas_precomputadas for fornecido, deve ser uma lista de dicts:
        {
            "ref": <dict do porto>,
            "distance_km": float,
            "duration_min": float,
            "geometry": [[lat, lon], ...]
        }
    """
    print("\n🗺️  [LOG] Criando mapa multi-ref...")
    print(f"[LOG] ORS API KEY presente? {'SIM' if ors_api_key else 'NÃO'}")
    if nm_fantasia_selecionado:
        print(f"[LOG] Empresa selecionada: {nm_fantasia_selecionado}")
    if porto_id_selecionado:
        print(f"[LOG] Porto filtrado: {porto_id_selecionado}")
    else:
        print("[LOG] Porto filtrado: TODOS")

    # Foco em Santa Catarina (centro aproximado)
    center_sc = (-27.0, -50.5)
    zoom_sc = 7

    # OpenStreetMap deixa cidades e estradas mais visíveis
    m = folium.Map(location=center_sc, zoom_start=zoom_sc, tiles="OpenStreetMap")

    # Marcadores dos portos
    for ref in referencias:
        folium.Marker(
            location=(ref["latitude"], ref["longitude"]),
            popup=f"<b>🚢 {ref['nome']}</b><br>{ref['sg_uf']}",
            icon=folium.Icon(color="darkblue", icon="anchor", prefix="fa"),
        ).add_to(m)

    # Colormap pela distância mínima (pontos mais escuros)
    vmin = df_distancias["distancia_km"].min()
    vmax = df_distancias["distancia_km"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn_r")

    # Empresa selecionada
    empresa_destacada = None
    if nm_fantasia_selecionado:
        mask_sel = df_distancias["nm_nome_fantasia"] == nm_fantasia_selecionado
        if mask_sel.any():
            empresa_destacada = df_distancias[mask_sel].iloc[0]
            print(f"[LOG] ✅ Match empresa selecionada: {empresa_destacada['nm_nome_fantasia']}")
        else:
            print(f"[LOG] ⚠️ Nenhum match para nm_nome_fantasia = {nm_fantasia_selecionado}")

    cluster = MarkerCluster(name="Empresas").add_to(m)

    # Pontos das empresas – mais escuros e com borda mais forte
    for _, row in df_distancias.iterrows():
        cor_rgb = cmap(norm(row["distancia_km"]))[:3]
        cor_hex = mcolors.rgb2hex(cor_rgb)

        is_selected = (
            empresa_destacada is not None and row["nu_cnpj"] == empresa_destacada["nu_cnpj"]
        )

        radius = 5 if not is_selected else 10
        weight = 1 if not is_selected else 3  # borda mais marcante
        opacity = 0.9 if not is_selected else 1.0  # mais opaco
        border_color = "#333333" if not is_selected else "#000000"  # contorno mais escuro

        popup_html = (
            f"{row['nm_nome_fantasia']} - {row['nm_mun']}/{row['sg_uf']} "
            f"({row['distancia_km']:.2f} km até {row['ref_mais_proxima_nome']})"
        )

        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=radius,
            popup=folium.Popup(popup_html, max_width=280),
            color=border_color,
            fill=True,
            fill_color=cor_hex,
            fill_opacity=opacity,
            weight=weight,
        ).add_to(cluster)

    # Rotas para a empresa selecionada
    if empresa_destacada is not None:
        coord_dest = (empresa_destacada["latitude"], empresa_destacada["longitude"])

        # Decide quais referências usar para rotas (todas ou apenas 1)
        if porto_id_selecionado:
            refs_para_rotas = [r for r in referencias if r["id"] == porto_id_selecionado]
        else:
            refs_para_rotas = referencias

        rotas_ok: List[Dict] = []

        # Se vieram rotas pré-computadas, reaproveita e filtra
        if rotas_precomputadas is not None:
            print("[LOG] Usando rotas pré-computadas no mapa...")
            for rota in rotas_precomputadas:
                ref = rota["ref"]
                if refs_para_rotas and not any(r["id"] == ref["id"] for r in refs_para_rotas):
                    continue
                if rota.get("geometry"):
                    rotas_ok.append(rota)

        # Se não houver pré-computadas, tenta chamar ORS (fallback)
        elif ors_api_key and refs_para_rotas:
            print("[LOG] Chamando ORS para referências selecionadas (fallback)...")
            for ref in refs_para_rotas:
                coord_ref = (ref["latitude"], ref["longitude"])
                rota = obter_rota_ors(coord_ref, coord_dest, api_key=ors_api_key)
                if rota and rota.get("geometry"):
                    rota["ref"] = ref
                    rotas_ok.append(rota)

        if rotas_ok:
            if porto_id_selecionado:
                # Só um porto → todas as rotas são "mínimas" por definição
                for rota in rotas_ok:
                    ref = rota["ref"]
                    folium.PolyLine(
                        locations=rota["geometry"],
                        color="#006400",  # verde escuro
                        weight=5,
                        opacity=0.9,
                        dash_array=None,
                        popup=(
                            f"Rota até {ref['nome']}: "
                            f"{rota['distance_km']:.1f} km, {rota['duration_min']:.1f} min"
                        ),
                    ).add_to(m)
            else:
                # Vários portos → destaque apenas a rota mínima em verde escuro
                rota_min = min(rotas_ok, key=lambda r: r["distance_km"])
                print(
                    f"[LOG] ✅ Rota viária mínima: {rota_min['ref']['nome']} "
                    f"({rota_min['distance_km']:.1f} km)"
                )

                for rota in rotas_ok:
                    ref = rota["ref"]
                    is_min = rota is rota_min

                    color = "#006400" if is_min else "#1f77b4"
                    weight = 5 if is_min else 3
                    dash_array = None if is_min else "6, 4"

                    folium.PolyLine(
                        locations=rota["geometry"],
                        color=color,
                        weight=weight,
                        opacity=0.9,
                        dash_array=dash_array,
                        popup=(
                            f"Rota até {ref['nome']}: "
                            f"{rota['distance_km']:.1f} km, {rota['duration_min']:.1f} min"
                        ),
                    ).add_to(m)
        else:
            # Fallback: linha reta
            print("[LOG] ⚠️ ORS indisponível ou sem rotas. Desenhando linhas retas.")
            linhas = []
            for ref in refs_para_rotas:
                coord_ref = (ref["latitude"], ref["longitude"])
                d_km = geodesic(coord_ref, coord_dest).km
                linhas.append((ref, coord_ref, d_km))

            if linhas:
                if porto_id_selecionado:
                    # Só uma linha, já é "mínima"
                    for ref, coord_ref, d_km in linhas:
                        folium.PolyLine(
                            locations=[coord_ref, coord_dest],
                            color="#006400",
                            weight=5,
                            opacity=0.9,
                            dash_array=None,
                            popup=f"Linha até {ref['nome']}: {d_km:.1f} km (reta)",
                        ).add_to(m)
                else:
                    # Várias linhas → destaca a menor
                    ref_min, coord_min, d_min = min(linhas, key=lambda x: x[2])
                    print(
                        f"[LOG] ✅ Menor linha reta: {ref_min['nome']} "
                        f"({d_min:.1f} km)"
                    )

                    for ref, coord_ref, d_km in linhas:
                        is_min = ref is ref_min
                        color = "#006400" if is_min else "#1f77b4"
                        weight = 5 if is_min else 3
                        dash_array = None if is_min else "6, 4"

                        folium.PolyLine(
                            locations=[coord_ref, coord_dest],
                            color=color,
                            weight=weight,
                            opacity=0.9,
                            dash_array=dash_array,
                            popup=f"Linha até {ref['nome']}: {d_km:.1f} km (reta)",
                        ).add_to(m)

    folium.LayerControl().add_to(m)
    print("   ✅ [LOG] Mapa multi-ref criado com sucesso")

    return m


#### PREVIEW DOS PRODUTOS

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


@dataclass
class FrameCheckResult:
    final_url: str
    decision: str  # "ALLOW" | "BLOCK" | "UNKNOWN"
    reason: str
    x_frame_options: str | None
    csp_frame_ancestors: str | None


def _get_origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _extract_frame_ancestors(csp: str | None) -> str | None:
    if not csp:
        return None
    parts = [p.strip() for p in csp.split(";")]
    for part in parts:
        if part.lower().startswith("frame-ancestors"):
            return part[len("frame-ancestors") :].strip() or ""
    return None


def _token_allows_embedding(token: str, app_origin: str, page_origin: str) -> bool:
    token = (token or "").strip()

    if token in ("'none'", "none"):
        return False
    if token in ("'self'", "self"):
        return app_origin == page_origin
    if token == "*":
        return True

    # scheme only (ex: https:)
    if token.endswith(":") and token[:-1] in ("http", "https"):
        return app_origin.startswith(token)

    # origin/host
    try:
        if "://" not in token:
            return urlparse(app_origin).netloc == token

        tok = urlparse(token)
        app = urlparse(app_origin)

        if tok.hostname and tok.hostname.startswith("*."):
            suffix = tok.hostname[1:]  # ".example.com"
            return (app.hostname or "").endswith(suffix) and (tok.scheme == app.scheme)

        return (tok.scheme == app.scheme) and (tok.netloc == app.netloc)
    except Exception:
        return False


def _csp_allows_embedding(frame_ancestors: str, app_origin: str, page_origin: str) -> tuple[bool, str]:
    tokens = [t for t in (frame_ancestors or "").split() if t.strip()]
    if not tokens:
        return False, "CSP frame-ancestors vazio (tratado como bloqueio)."

    if "'none'" in tokens or "none" in tokens:
        return False, "CSP frame-ancestors 'none' (bloqueia iframes)."

    for t in tokens:
        if _token_allows_embedding(t, app_origin=app_origin, page_origin=page_origin):
            return True, f"CSP frame-ancestors permite: {t}"

    return False, "CSP frame-ancestors não inclui o origin do app."


def infer_app_origin(default_local: str = "http://localhost:8501") -> str:
    """
    Melhor esforço:
    - Se existir APP_ORIGIN em st.secrets, usa ele.
    - Senão, usa localhost (bom pra dev).
    """
    try:
        v = st.secrets.get("APP_ORIGIN", None)
        if v:
            return str(v).strip()
    except Exception:
        pass
    return default_local


@st.cache_data(ttl=60 * 60, show_spinner=False)
def check_iframe_allowed(url: str, app_origin: str, timeout_s: int = 10) -> FrameCheckResult:
    session = requests.Session()

    def _req(method: str):
        return session.request(
            method=method,
            url=url,
            headers=DEFAULT_HEADERS,
            timeout=timeout_s,
            allow_redirects=True,
        )

    resp = None
    # 1) HEAD
    try:
        resp = _req("HEAD")
        if resp.status_code >= 400 or not resp.headers:
            raise RuntimeError(f"HEAD status {resp.status_code}")
    except Exception:
        # 2) GET fallback
        try:
            resp = session.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=timeout_s,
                allow_redirects=True,
                stream=True,
            )
        except Exception as e:
            return FrameCheckResult(
                final_url=url,
                decision="UNKNOWN",
                reason=f"Falha ao acessar URL para checar headers: {e}",
                x_frame_options=None,
                csp_frame_ancestors=None,
            )

    final_url = resp.url or url
    page_origin = _get_origin(final_url)

    xfo = resp.headers.get("X-Frame-Options") or resp.headers.get("x-frame-options")
    csp = resp.headers.get("Content-Security-Policy") or resp.headers.get("content-security-policy")

    xfo_norm = xfo.strip().lower() if isinstance(xfo, str) else None

    # X-Frame-Options
    if xfo_norm:
        if "deny" in xfo_norm:
            return FrameCheckResult(final_url, "BLOCK", "X-Frame-Options: DENY", xfo, None)
        if "sameorigin" in xfo_norm and app_origin != page_origin:
            return FrameCheckResult(final_url, "BLOCK", "X-Frame-Options: SAMEORIGIN", xfo, None)

    # CSP frame-ancestors
    fa = _extract_frame_ancestors(csp if isinstance(csp, str) else None)
    if fa is not None:
        allowed, why = _csp_allows_embedding(fa, app_origin=app_origin, page_origin=page_origin)
        if not allowed:
            return FrameCheckResult(final_url, "BLOCK", why, xfo, fa)

    # Sem bloqueio explícito encontrado
    if xfo_norm or fa is not None or csp:
        return FrameCheckResult(final_url, "ALLOW", "Sem bloqueio explícito por XFO/CSP.", xfo, fa)

    return FrameCheckResult(final_url, "UNKNOWN", "Sem XFO/CSP nos headers (não dá pra garantir).", xfo, fa)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Gerando preview...") 
def get_page_screenshot_bytes(url: str, timeout_ms: int = 20000) -> Optional[bytes]:
    """
    Tira screenshot da página via Playwright.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        st.error("Playwright não instalado. Execute: pip install playwright && playwright install chromium")
        return None

    try:
        with sync_playwright() as p:
            # Lançando o browser
            browser = p.chromium.launch(
                headless=True, 
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            
            # Contexto com User Agent de um navegador real para evitar bloqueios simples
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            
            page = context.new_page()
            
            # Tenta carregar a página
            # Mudamos de 'networkidle' para 'domcontentloaded' ou 'load' porque 
            # sites com muitos rastreadores nunca chegam em 'networkidle' e dão timeout.
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                # Espera um tempinho extra para renderizar imagens dinâmicas
                page.wait_for_timeout(2000) 
            except Exception as e:
                print(f"Aviso no carregamento: {e}")
                # Se der timeout no load, tentamos tirar o print assim mesmo do que carregou

            png_bytes = page.screenshot(full_page=False, type="png")
            
            context.close()
            browser.close()
            return png_bytes
            
    except Exception as e:
        # Isso ajuda a debugar no console do Streamlit
        print(f"Erro ao capturar screenshot de {url}:")
        traceback.print_exc()
        return None
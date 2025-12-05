"""
Módulos para análise geoespacial de distâncias a partir de Caçador/SC
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import matplotlib.colors as mcolors
from typing import Dict, Tuple, Optional
import requests


class CalculadoraDistanciasAvancada:
    """
    Calcula distâncias a partir de Caçador/SC como referência.
    """

    def __init__(self, df_empresas: pd.DataFrame, municipio_ref: Dict):
        print("\n" + "="*60)
        print("🧮 [LOG] INICIALIZANDO CALCULADORA DE DISTÂNCIAS")
        print("="*60)

        self.municipio_ref = municipio_ref
        self.coord_ref = (municipio_ref['latitude'], municipio_ref['longitude'])

        # Limpar dados
        self.df = df_empresas.dropna(subset=['latitude', 'longitude']).copy()

        registros_removidos = len(df_empresas) - len(self.df)

        print(f"📍 [LOG] Ponto de referência: {municipio_ref['nm_mun']}/{municipio_ref['sg_uf']}")
        print(f"   [LOG] Coordenadas: ({municipio_ref['latitude']:.6f}, {municipio_ref['longitude']:.6f})")
        print(f"\n📊 [LOG] Dados válidos:")
        print(f"   ✅ [LOG] Pontos com coordenadas: {len(self.df):,}")
        print(f"   ❌ [LOG] Pontos removidos: {registros_removidos:,}")
        print("="*60 + "\n")

    def calcular_todas_distancias(self) -> pd.DataFrame:
        """
        Calcula distâncias de Caçador para todas as empresas.
        """
        print("🔄 [LOG] Calculando distâncias de Caçador/SC para todos os pontos...")
        print(f"   [LOG] Total de cálculos: {len(self.df):,}")

        distancias = []
        total = len(self.df)

        for idx, row in self.df.iterrows():
            coord_destino = (row['latitude'], row['longitude'])
            dist_km = geodesic(self.coord_ref, coord_destino).km

            # Verificar se é o próprio município de referência
            eh_cacador = row['cd_mun_ibge'] == self.municipio_ref['cd_mun_ibge']

            distancias.append({
                'nu_cnpj': row['nu_cnpj'],
                'nm_nome_fantasia': row['nm_nome_fantasia'],
                'nm_razao_social': row['nm_razao_social'],
                'nm_mun': row['nm_mun'],
                'sg_uf': row['sg_uf'],
                'cd_mun_ibge': row['cd_mun_ibge'],
                'nm_porte_obs': row.get('nm_porte_obs', ''),
                'cd_cnae_fiscal_principal': row.get('cd_cnae_fiscal_principal', ''),
                'nm_cnae_fiscal_principal': row.get('nm_cnae_fiscal_principal', ''),
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'distancia_km': dist_km,
                'eh_cacador': eh_cacador
            })

            # Log de progresso
            if (len(distancias)) % 20 == 0:
                print(f"   ⏳ [LOG] Progresso: {len(distancias)}/{total} ({len(distancias)/total*100:.1f}%)")

        df_dist = pd.DataFrame(distancias).sort_values('distancia_km')

        print(f"\n✅ [LOG] Cálculo concluído!")
        print(f"   • [LOG] Total de distâncias calculadas: {len(df_dist):,}")
        print(f"   • [LOG] Empresas em Caçador/SC: {df_dist['eh_cacador'].sum()}")

        if len(df_dist[~df_dist['eh_cacador']]) > 0:
            print(f"   • [LOG] Distância mínima: {df_dist[~df_dist['eh_cacador']]['distancia_km'].min():.2f} km")
            print(f"   • [LOG] Distância máxima: {df_dist['distancia_km'].max():.2f} km")

        return df_dist

    def extrair_kpis_completos(self, df_distancias: pd.DataFrame) -> Dict:
        """
        Extrai KPIs detalhados de distância.
        """
        print("\n📊 [LOG] Extraindo KPIs...")

        # Separar empresas de Caçador das demais
        df_cacador = df_distancias[df_distancias['eh_cacador']].copy()
        df_fora = df_distancias[~df_distancias['eh_cacador']].copy()

        if len(df_fora) == 0:
            print("⚠️  [LOG] Apenas empresas de Caçador encontradas")
            return {"erro": "Sem empresas fora de Caçador"}

        # Helper para tratar valores nulos
        def safe_get(row, col, default="(Não informado)"):
            val = row[col]
            if pd.isna(val) or val == "":
                return default
            return str(val)

        # Pegar primeira e última linha (já ordenado por distância)
        row_proximo = df_fora.iloc[0]
        row_distante = df_fora.iloc[-1]

        kpis = {
            # Informações de referência
            'referencia': {
                'municipio': self.municipio_ref['nm_mun'],
                'uf': self.municipio_ref['sg_uf'],
                'empresas_local': len(df_cacador)
            },

            # Extremos (fora de Caçador)
            'mais_proximo': {
                'cnpj': safe_get(row_proximo, 'nu_cnpj'),
                'nome_fantasia': safe_get(row_proximo, 'nm_nome_fantasia'),
                'razao_social': safe_get(row_proximo, 'nm_razao_social'),
                'municipio': safe_get(row_proximo, 'nm_mun'),
                'uf': safe_get(row_proximo, 'sg_uf', 'BR'),
                'distancia_km': row_proximo['distancia_km']
            },
            'mais_distante': {
                'cnpj': safe_get(row_distante, 'nu_cnpj'),
                'nome_fantasia': safe_get(row_distante, 'nm_nome_fantasia'),
                'razao_social': safe_get(row_distante, 'nm_razao_social'),
                'municipio': safe_get(row_distante, 'nm_mun'),
                'uf': safe_get(row_distante, 'sg_uf', 'BR'),
                'distancia_km': row_distante['distancia_km']
            },

            # Estatísticas descritivas
            'estatisticas': {
                'media_km': df_fora['distancia_km'].mean(),
                'mediana_km': df_fora['distancia_km'].median(),
                'desvio_padrao_km': df_fora['distancia_km'].std(),
                'p25_km': df_fora['distancia_km'].quantile(0.25),
                'p75_km': df_fora['distancia_km'].quantile(0.75),
                'p90_km': df_fora['distancia_km'].quantile(0.90)
            },

            # Distribuição por faixas
            'distribuicao': {
                'ate_50km': (df_fora['distancia_km'] <= 50).sum(),
                'de_50_a_100km': ((df_fora['distancia_km'] > 50) & (df_fora['distancia_km'] <= 100)).sum(),
                'de_100_a_200km': ((df_fora['distancia_km'] > 100) & (df_fora['distancia_km'] <= 200)).sum(),
                'de_200_a_500km': ((df_fora['distancia_km'] > 200) & (df_fora['distancia_km'] <= 500)).sum(),
                'acima_500km': (df_fora['distancia_km'] > 500).sum()
            },

            # Totais
            'totais': {
                'total_pontos': len(df_distancias),
                'pontos_fora_cacador': len(df_fora),
                'pontos_em_cacador': len(df_cacador)
            }
        }

        print("✅ [LOG] KPIs extraídos com sucesso")
        print(f"   • [LOG] Mais próximo: {kpis['mais_proximo']['nome_fantasia']} ({kpis['mais_proximo']['distancia_km']:.2f} km)")
        print(f"   • [LOG] Mais distante: {kpis['mais_distante']['nome_fantasia']} ({kpis['mais_distante']['distancia_km']:.2f} km)")

        return kpis



class AnalisadorClusters:
    """
    Identifica agrupamentos geográficos de empresas.
    """

    def __init__(self, df_distancias: pd.DataFrame):
        self.df = df_distancias.copy()
        print("\n" + "="*60)
        print("🔍 [LOG] INICIALIZANDO ANÁLISE DE CLUSTERS")
        print("="*60)
        print(f"   [LOG] Pontos para análise: {len(self.df):,}")

    def clustering_dbscan(self, raio_km: float = 100, min_pontos: int = 2) -> pd.DataFrame:
        """
        Clustering baseado em densidade (DBSCAN).
        """
        print(f"\n🎯 [LOG] Executando DBSCAN...")
        print(f"   [LOG] Parâmetros: Raio={raio_km} km, Min pontos={min_pontos}")

        # Preparar coordenadas
        coords = self.df[['latitude', 'longitude']].values

        # Converter raio para radianos
        kms_per_radian = 6371.0088
        epsilon = raio_km / kms_per_radian

        # Executar DBSCAN
        db = DBSCAN(eps=epsilon, min_samples=min_pontos, metric='haversine')
        self.df['cluster_dbscan'] = db.fit_predict(np.radians(coords))

        # Análise dos clusters
        n_clusters = len(set(self.df['cluster_dbscan'])) - (1 if -1 in self.df['cluster_dbscan'] else 0)
        n_noise = (self.df['cluster_dbscan'] == -1).sum()

        print(f"\n📊 [LOG] Resultados DBSCAN:")
        print(f"   ✅ [LOG] Clusters identificados: {n_clusters}")
        print(f"   ⚪ [LOG] Pontos isolados: {n_noise}")

        if n_clusters > 0:
            for cluster_id in sorted(self.df['cluster_dbscan'].unique()):
                if cluster_id == -1:
                    continue

                df_cluster = self.df[self.df['cluster_dbscan'] == cluster_id]
                municipios = df_cluster['nm_mun'].value_counts()

                print(f"\n   🔵 [LOG] Cluster {cluster_id}:")
                print(f"      • [LOG] Empresas: {len(df_cluster)}")
                print(f"      • [LOG] Principal município: {municipios.index[0]} ({municipios.iloc[0]} empresas)")
                print(f"      • [LOG] Distância média: {df_cluster['distancia_km'].mean():.1f} km")

        return self.df

    def clustering_kmeans(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Clustering K-Means.
        """
        print(f"\n🎯 [LOG] Executando K-Means com {n_clusters} clusters...")

        features = self.df[['latitude', 'longitude', 'distancia_km']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster_kmeans'] = kmeans.fit_predict(features_scaled)

        print(f"\n📊 [LOG] Resultados K-Means:")
        for cluster_id in range(n_clusters):
            df_cluster = self.df[self.df['cluster_kmeans'] == cluster_id]
            print(f"   🔵 [LOG] Cluster {cluster_id}: {len(df_cluster)} empresas, dist média {df_cluster['distancia_km'].mean():.1f} km")

        return self.df

    def analise_completa(self, raio_dbscan: float = 100) -> pd.DataFrame:
        """
        Executa ambos os métodos de clustering.
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
    profile: str = "driving-car",
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

    # Endpoint em GeoJSON (compatível com o resto do seu código)
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"

    # 🔁 Converte (lat, lon) -> [lon, lat] para a ORS
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

        # Loga corpo de erro se não for 200
        if resp.status_code != 200:
            print("[ORS] Corpo da resposta (erro/troubleshooting):")
            print(resp.text[:800])  # limita pra não floodar

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




def criar_mapa_distancias_cacador(
    df_distancias: pd.DataFrame,
    municipio_ref: dict,
    kpis: dict,
    nm_fantasia_selecionado: Optional[str] = None,
    ors_api_key: Optional[str] = None
) -> folium.Map:
    """
    Cria mapa Folium com:
      - Caçador como referência
      - pontos coloridos pela distância
      - empresa selecionada destacada
      - rota rodoviária ORS (se disponível) ou linha reta como fallback
    """
    print("\n🗺️  [LOG] Criando mapa...")
    print(f"[LOG] ORS API KEY presente? {'SIM' if ors_api_key else 'NÃO'}")
    if nm_fantasia_selecionado:
        print(f"[LOG] Empresa selecionada: {nm_fantasia_selecionado}")

    coord_ref = (municipio_ref['latitude'], municipio_ref['longitude'])
    df_fora = df_distancias[~df_distancias['eh_cacador']].copy()
    if df_fora.empty:
        df_fora = df_distancias.copy()

    m = folium.Map(location=coord_ref, zoom_start=6, tiles="CartoDB positron")

    # Referência Caçador
    folium.Marker(
        location=coord_ref,
        popup=f"<b>📍 Referência</b><br>{municipio_ref['nm_mun']}/{municipio_ref['sg_uf']}",
        icon=folium.Icon(color="blue", icon="star", prefix='fa')
    ).add_to(m)

    # Anéis fixos
    raios_fixos_km = [50, 150, 300, 500]
    cores_raios = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    labels_raios = [
        '50 km de Caçador',
        '150 km de Caçador',
        '300 km de Caçador',
        '500 km de Caçador'
    ]
    for raio, cor, label in zip(raios_fixos_km, cores_raios, labels_raios):
        folium.Circle(
            location=coord_ref,
            radius=raio * 1000,
            color=cor,
            fill=False,
            weight=2,
            opacity=0.5,
            popup=label
        ).add_to(m)

    # Colormap
    vmin = df_fora['distancia_km'].min()
    vmax = df_fora['distancia_km'].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('RdYlGn_r')

    # Empresa selecionada
    empresa_destacada = None
    if nm_fantasia_selecionado:
        mask_sel = df_distancias['nm_nome_fantasia'] == nm_fantasia_selecionado
        if mask_sel.any():
            empresa_destacada = df_distancias[mask_sel].iloc[0]
            print(f"[LOG] ✅ Match empresa selecionada: {empresa_destacada['nm_nome_fantasia']}")
        else:
            print(f"[LOG] ⚠️ Nenhum match para nm_nome_fantasia = {nm_fantasia_selecionado}")

    cluster = MarkerCluster(name="Empresas").add_to(m)

    # Pontos
    for _, row in df_distancias.iterrows():
        cor_hex = "#999999"
        if not row['eh_cacador']:
            cor_rgb = cmap(norm(row['distancia_km']))[:3]
            cor_hex = mcolors.rgb2hex(cor_rgb)

        is_selected = (
            empresa_destacada is not None and
            row['nu_cnpj'] == empresa_destacada['nu_cnpj']
        )

        radius = 5 if not is_selected else 10
        weight = 1 if not is_selected else 4
        opacity = 0.7 if not is_selected else 1.0
        border_color = cor_hex if not is_selected else "#1f77b4"

        popup_html = f"{row['nm_nome_fantasia']} - {row['nm_mun']}/{row['sg_uf']} ({row['distancia_km']:.2f} km)"

        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=radius,
            popup=folium.Popup(popup_html, max_width=280),
            color=border_color,
            fill=True,
            fill_color=cor_hex,
            fill_opacity=opacity,
            weight=weight
        ).add_to(cluster)

    # Rota entre Caçador e empresa selecionada
    if empresa_destacada is not None:
        coord_dest = (empresa_destacada['latitude'], empresa_destacada['longitude'])

        rota = None
        if ors_api_key:
            print("[LOG] Chamando ORS para rota rodoviária...")
            rota = obter_rota_ors(coord_ref, coord_dest, api_key=ors_api_key)
        else:
            print("[LOG] ORS_API_KEY ausente. Usando linha reta.")

        if rota is not None and rota.get("geometry"):
            # Rota rodoviária
            folium.PolyLine(
                locations=rota["geometry"],
                color="#1f77b4",
                weight=4,
                opacity=0.9,
                dash_array=None,
                popup=f"Rota rodoviária: {rota['distance_km']:.1f} km, {rota['duration_min']:.1f} min"
            ).add_to(m)
            print("[LOG] ✅ Rota ORS desenhada no mapa.")
        else:
            # Fallback: linha reta
            folium.PolyLine(
                locations=[coord_ref, coord_dest],
                color="#1f77b4",
                weight=3,
                opacity=0.8,
                dash_array="8, 4",
                popup=f"Rota (reta): {empresa_destacada['distancia_km']:.2f} km"
            ).add_to(m)
            print("[LOG] ⚠️ Não foi possível obter rota ORS. Desenhando linha reta.")

    folium.LayerControl().add_to(m)

    print("   ✅ [LOG] Mapa criado com sucesso")

    return m


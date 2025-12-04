import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
from pathlib import Path

######## Configurando o caminho para a pasta raiz do projeto ########
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p

dados = pd.read_excel(_resolve_path("Dados/Brutos/dados.xlsx"))
mapa = gpd.read_file(_resolve_path("Dados/Brutos/BR_Municipios_2024.shp"))

mapa = mapa[mapa["SIGLA_UF"].isin(["SC"])]
mapa["geometry"] = mapa["geometry"].simplify(0.01)

# Merge: todos os municípios da região Sul
mapa_merged = mapa.merge(
    dados,
    left_on="NM_MUN",
    right_on="munic",
    how="left"  # 'left' para manter todos os municípios do mapa
)

mapa_merged.head()

lat_long = pd.read_csv(_resolve_path("Referências/dim_municipio_br.csv"))

lat_long.head()

lat_long = lat_long[['nm_municipio_1mai_ca',
                     'nu_latitude',
                     'nu_longitude']]

mapa_merged = mapa_merged.merge(
    lat_long,
    left_on="NM_MUN",
    right_on="nm_municipio_1mai_ca",
    how="left"
)

geojson_data = mapa_merged.to_json()

with open(_resolve_path("Dados/Processados/eucalipto_sc.geojson"), "w", encoding="utf-8") as f:
    f.write(geojson_data)
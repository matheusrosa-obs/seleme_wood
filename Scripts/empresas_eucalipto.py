import pandas as pd
from pathlib import Path

######## Configurando o caminho para a pasta raiz do projeto ########
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p

empresas = pd.read_csv(_resolve_path("Dados/Brutos/empresas_eucalipto.csv"))

empresas = empresas[empresas["sg_uf"] == "SC"]

empresas.head()

#empresas['cd_mun_ibge'] = empresas['cd_mun_ibge'].astype(str)

lat_long = pd.read_csv(_resolve_path("Referências/dim_municipio_br.csv"))

lat_long.head()

lat_long = lat_long[['cd_mun_7d',
                     'nm_uf',
                     'nm_municipio_1mai_sa',
                     'nu_latitude',
                     'nu_longitude']]

empresas = empresas.merge(
    lat_long,
    left_on="cd_mun_ibge",
    right_on="cd_mun_7d",
    how="left"
)

empresas = empresas[['nm_razao_social', 'nm_nome_socio', 'dt_inicio_atividade',
                     'nm_municipio_1mai_sa', 'endereco_completo', 'nu_telefone_1',
                     'nm_correio_eletronico', 'nu_latitude', 'nu_longitude'
                     ]]

empresas['nm_razao_social'] = empresas['nm_razao_social'].str.title()
empresas['nm_nome_socio'] = empresas['nm_nome_socio'].str.title()

empresas.head()
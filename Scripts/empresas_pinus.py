import pandas as pd
from pathlib import Path

######## Configurando o caminho para a pasta raiz do projeto ########
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p

empresas = pd.read_excel(_resolve_path("Dados/Brutos/empresas_pinus.xlsx"))

empresas.head()

empresas_munic = empresas.groupby(
    ['nm_mun', 'cd_mun_ibge', 'sg_uf', 'nm_cnae_fiscal_principal']
).size().reset_index(name='quantidade_empresas')

lat_long = pd.read_csv(_resolve_path("Referências/dim_municipio_br.csv"))

lat_long.head()

lat_long = lat_long[['cd_mun_7d',
                     'nm_uf',
                     'nm_municipio_1mai_ca',
                     'nm_microrregiao',
                     'nu_latitude',
                     'nu_longitude']]

empresas_munic = empresas_munic.merge(
    lat_long,
    left_on=["cd_mun_ibge"],
    right_on=["cd_mun_7d"],
    how="left"
)

empresas_munic.head()

empresas_munic.to_csv(_resolve_path("Dados/Processados/empresas_pinus_munic.csv"), index=False)
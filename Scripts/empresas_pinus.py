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
    ['nm_municipio_1mai_sa', 'cd_mun_7d', 'nm_uf', 'sg_uf', 'nm_cnae_fiscal_principal']
).size().reset_index(name='quantidade_empresas')

print(empresas['nm_cnae_fiscal_principal'].unique())

# Mapeando os setores para os dois grupos desejados
setores_map = {
    'Comércio varejista de materiais de construção não especificados anteriormente': 'Comércio varejista de materiais de construção',
    'Comércio varejista de materiais de construção em geral': 'Comércio varejista de materiais de construção',
    'Comércio atacadista de madeira e produtos derivados': 'Comércio atacadista de materiais de construção',
    'Comércio varejista de madeira e artefatos': 'Comércio varejista de materiais de construção'
}

empresas['grupo_setor'] = empresas['nm_cnae_fiscal_principal'].map(setores_map).fillna(empresas['nm_cnae_fiscal_principal'])

# Exemplo de agrupamento usando o novo grupo
empresas_munic_grupo = empresas.groupby(
    ['nm_municipio_1mai_sa', 'cd_mun_7d', 'nm_uf', 'sg_uf', 'grupo_setor']
).size().reset_index(name='quantidade_empresas')

lat_long = pd.read_csv(_resolve_path("Referências/dim_municipio_br.csv"))

lat_long.head()

lat_long = lat_long[['cd_mun_7d',
                     'nm_uf',
                     'nm_municipio_1mai_sa',
                     'nm_microrregiao',
                     'nu_latitude',
                     'nu_longitude']]

empresas_munic_grupo = empresas_munic_grupo.merge(
    lat_long,
    left_on=["cd_mun_7d"],
    right_on=["cd_mun_7d"],
    how="left"
)

empresas_munic_grupo.head()

empresas_munic_grupo.to_csv(_resolve_path("Dados/Processados/empresas_pinus_munic.csv"), index=False)
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
                     'nm_microrregiao']]

empresas = empresas.merge(
    lat_long,
    left_on="cd_mun_ibge",
    right_on="cd_mun_7d",
    how="left"
)

empresas['ano_inicio'] = empresas['dt_inicio_atividade'].astype(str).str[:4]

empresas = empresas[['nm_razao_social', 'nm_porte_obs', 'ano_inicio',
                     'nm_microrregiao', 'nm_municipio_1mai_sa', 'endereco_completo', 'nu_telefone_1',
                     'nm_correio_eletronico'
                     ]]

empresas['nm_razao_social'] = empresas['nm_razao_social'].str.title()
empresas['endereco_completo'] = empresas['endereco_completo'].str.title()

empresas = empresas.rename(columns={
    "nm_razao_social": "Razão Social",
    "nm_porte_obs": "Porte da Empresa",
    "ano_inicio": "Ano de Início",
    "nm_microrregiao": "Microrregião",
    "nm_municipio_1mai_sa": "Município",
    "endereco_completo": "Endereço Completo",
    "nu_telefone_1": "Telefone",
    "nm_correio_eletronico": "E-mail"
})

empresas['Telefone'] = empresas['Telefone'].astype(str).str.replace(r'^(\d{4})(\d+)$', r'\1-\2', regex=True)

empresas = empresas.sort_values(by="Razão Social")

empresas.head()

empresas.to_csv(_resolve_path("Dados/Processados/empresas_eucalipto_sc.csv"), index=False)
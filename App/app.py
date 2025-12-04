#################################################
import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path

from warnings import filterwarnings
import re
filterwarnings("ignore")

######## Configurando o caminho para a pasta raiz do projeto ########
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _project_root() / p


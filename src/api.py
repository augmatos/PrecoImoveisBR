"""
API REST (FastAPI) que serve os modelos do projeto num único endpoint.

A partir das características de um imóvel, devolve:
- **preço estimado** (modelo de regressão Gradient Boosting);
- **segmento de mercado** (K-Means, usando o preço previsto);
- **yield de aluguel** do bairro (retorno bruto anual).

Rodar (após `python src/treinar_modelos.py`):

    uvicorn src.api:app --reload
    # docs interativas em http://localhost:8000/docs
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS = Path(__file__).resolve().parent.parent / "models"

# --- carga dos artefatos (uma vez, na subida da API) ---
modelo_preco = joblib.load(MODELS / "modelo_preco.joblib")
seg = joblib.load(MODELS / "modelo_segmento.joblib")
META = json.loads((MODELS / "metadados.json").read_text(encoding="utf-8"))
BAIRROS = META["bairros_conhecidos"]

app = FastAPI(
    title="API — Mercado Imobiliário de Palmas/TO",
    description="Estima preço, segmento de mercado e yield de aluguel de um imóvel.",
    version="1.0.0",
)


class Imovel(BaseModel):
    tipo: str = Field("casa", examples=["casa"], description="casa, apartamento, etc.")
    area_util: float = Field(..., gt=0, examples=[120], description="Área útil em m²")
    quartos: int = Field(..., ge=0, examples=[3])
    banheiros: int = Field(..., ge=0, examples=[2])
    vagas: int = Field(..., ge=0, examples=[2])
    bairro: str = Field(..., examples=["Plano Diretor Sul"])


class Previsao(BaseModel):
    preco_estimado: float
    preco_m2_estimado: float
    segmento: str
    yield_bairro_pct: float
    yield_e_da_cidade: bool
    bairro_conhecido: bool


def _segmentar(preco: float, im: Imovel) -> str:
    preco_m2 = preco / im.area_util
    linha = pd.DataFrame([{
        "preco": preco, "area_util": im.area_util, "quartos": im.quartos,
        "banheiros": im.banheiros, "vagas": im.vagas, "preco_m2": preco_m2,
    }])[seg["feats"]]
    X = seg["scaler"].transform(np.log1p(linha))
    cluster = int(seg["kmeans"].predict(X)[0])
    return seg["mapa"][str(cluster)] if str(cluster) in seg["mapa"] else seg["mapa"][cluster]


@app.get("/")
def raiz():
    return {"status": "ok", "docs": "/docs", "endpoints": ["/prever", "/bairros"]}


@app.get("/bairros")
def bairros():
    """Bairros conhecidos pelo modelo e yield disponível por bairro."""
    return {"bairros": BAIRROS, "yield": META["yield"]}


@app.post("/prever", response_model=Previsao)
def prever(imovel: Imovel):
    bairro_grp = imovel.bairro if imovel.bairro in BAIRROS else "Outros"
    X = pd.DataFrame([{
        "area_util": imovel.area_util, "quartos": imovel.quartos,
        "banheiros": imovel.banheiros, "vagas": imovel.vagas,
        "tipo": imovel.tipo, "bairro_grp": bairro_grp,
    }])
    preco = float(np.expm1(modelo_preco.predict(X)[0]))
    segmento = _segmentar(preco, imovel)

    yld = META["yield"]["por_bairro"].get(imovel.bairro)
    da_cidade = yld is None
    if da_cidade:
        yld = META["yield"]["cidade"]

    return Previsao(
        preco_estimado=round(preco, 2),
        preco_m2_estimado=round(preco / imovel.area_util, 2),
        segmento=segmento,
        yield_bairro_pct=yld,
        yield_e_da_cidade=da_cidade,
        bairro_conhecido=imovel.bairro in BAIRROS,
    )

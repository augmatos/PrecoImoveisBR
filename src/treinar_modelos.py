"""
Treina e serializa os artefatos usados pela API:

1. Modelo de **preço** (Gradient Boosting) — prevê o preço de venda.
2. Modelo de **segmentação** (StandardScaler + K-Means, k=4) — classifica o imóvel
   numa faixa de mercado a partir das features + preço previsto.
3. Tabela de **yield** por bairro — retorno bruto anual de aluguel.

Os artefatos são salvos em models/ (joblib + json). Rode após coletar os dados:

    python src/scraper.py --operacao venda   --max-paginas 100
    python src/scraper.py --operacao aluguel --max-paginas 100
    python src/treinar_modelos.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RESIDENCIAIS = ["casa", "apartamento", "cobertura", "flat", "casa em condominio", "sobrado"]
FEAT_NUM = ["area_util", "quartos", "banheiros", "vagas"]
FEAT_CAT = ["tipo", "bairro_grp"]
SEG_FEATS = ["preco", "area_util", "quartos", "banheiros", "vagas", "preco_m2"]
SEG_NOMES = ["Compacto / Entrada", "Padrão Médio", "Médio-Alto / Amplo", "Alto Padrão"]


def _csv_mais_recente(operacao: str) -> str:
    return max(glob.glob(str(RAW / f"imoveis_{operacao}_*.csv")), key=os.path.getmtime)


def carregar_limpo(operacao: str) -> pd.DataFrame:
    df = pd.read_csv(_csv_mais_recente(operacao))
    df = df[df["tipo"].isin(RESIDENCIAIS)].dropna(subset=["preco", "area_util"]).copy()
    df = df[(df["area_util"] >= 20) & (df["area_util"] <= 2000)]
    df["preco_m2"] = df["preco"] / df["area_util"]
    if operacao == "venda":
        df = df[(df["preco"] >= 50_000) & (df["preco"] <= 10_000_000)]
        df = df[df["preco_m2"].between(500, 30_000)]
        df = df.dropna(subset=["quartos"])
        df["vagas"] = df["vagas"].fillna(0)
        df["banheiros"] = df["banheiros"].fillna(df["banheiros"].median())
    else:
        df = df[(df["preco"] >= 300) & (df["preco"] <= 50_000)]
        df = df[df["preco_m2"].between(5, 300)]
    return df


def treinar_preco(venda: pd.DataFrame, bairros_freq: list[str]) -> Pipeline:
    df = venda.copy()
    df["bairro_grp"] = df["bairro"].where(df["bairro"].isin(bairros_freq), "Outros")
    X = df[FEAT_NUM + FEAT_CAT]
    y = np.log1p(df["preco"])
    pre = ColumnTransformer([
        ("num", StandardScaler(), FEAT_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEAT_CAT),
    ])
    pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=42))])
    pipe.fit(X, y)
    return pipe


def treinar_segmentacao(venda: pd.DataFrame):
    """Retorna (scaler, kmeans, mapa cluster->nome) ordenado por preço mediano."""
    X = np.log1p(venda[SEG_FEATS])
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler.transform(X))
    labels = km.labels_
    medianas = pd.Series(venda["preco"].values).groupby(labels).median().sort_values()
    mapa = {int(c): SEG_NOMES[i] for i, c in enumerate(medianas.index)}
    return scaler, km, mapa


def tabela_yield(venda: pd.DataFrame, aluguel: pd.DataFrame) -> dict:
    vb = venda.groupby("bairro").agg(venda_m2=("preco_m2", "median"), n_v=("preco", "size"))
    ab = aluguel.groupby("bairro").agg(aluguel_m2=("preco_m2", "median"), n_a=("preco", "size"))
    y = vb.join(ab, how="inner")
    y = y[(y["n_v"] >= 5) & (y["n_a"] >= 3)]
    y["yield_anual_pct"] = (y["aluguel_m2"] * 12) / y["venda_m2"] * 100
    cidade = (aluguel["preco_m2"].median() * 12) / venda["preco_m2"].median() * 100
    return {
        "por_bairro": {b: round(v, 2) for b, v in y["yield_anual_pct"].items()},
        "cidade": round(float(cidade), 2),
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    venda = carregar_limpo("venda")
    aluguel = carregar_limpo("aluguel")
    print(f"Venda: {len(venda)} | Aluguel: {len(aluguel)}")

    freq = venda["bairro"].value_counts()
    bairros_freq = freq[freq >= 10].index.tolist()

    preco = treinar_preco(venda, bairros_freq)
    scaler, km, mapa = treinar_segmentacao(venda)
    ydict = tabela_yield(venda, aluguel)

    joblib.dump(preco, MODELS / "modelo_preco.joblib")
    joblib.dump({"scaler": scaler, "kmeans": km, "mapa": mapa, "feats": SEG_FEATS},
                MODELS / "modelo_segmento.joblib")
    meta = {
        "yield": ydict,
        "bairros_conhecidos": sorted(bairros_freq),
        "tipos": RESIDENCIAIS,
        "segmentos": SEG_NOMES,
    }
    (MODELS / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("Artefatos salvos em models/:")
    print(" - modelo_preco.joblib")
    print(" - modelo_segmento.joblib")
    print(" - metadados.json")
    print(f"Yield calculado para {len(ydict['por_bairro'])} bairros (cidade: {ydict['cidade']}%)")


if __name__ == "__main__":
    main()

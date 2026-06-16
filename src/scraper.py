"""
Scraper de anúncios de imóveis à venda do Chaves na Mão.

Coleta, por cidade, os imóveis listados nas páginas de busca e salva um CSV
bruto em data/raw/. Os dados extraídos (preço, área, quartos, banheiros, vagas,
bairro) alimentam o projeto de regressão de preços de imóveis.

Boas práticas adotadas:
- Verifica o robots.txt antes de coletar (a paginação ?pg= é permitida).
- Define um User-Agent honesto e um intervalo entre requisições (rate limiting).
- Trata erros de rede com retries simples e segue em frente sem derrubar a coleta.

Uso:
    python src/scraper.py --uf to --cidade palmas --max-paginas 30
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.robotparser
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.chavesnamao.com.br"
LISTA_URL = BASE + "/imoveis-a-venda/{uf}-{cidade}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
DELAY_SEGUNDOS = 2.0  # intervalo educado entre páginas
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# campos numéricos extraídos dos "features" do card, por palavra-chave
FEATURE_KEYS = {
    "quartos": "quartos",
    "banheiros": "banheiros",
    "garagens": "vagas",
    "salas": "salas",
}


def pode_coletar(url: str) -> bool:
    """Consulta o robots.txt do site para a URL e o User-Agent usados.

    O robots.txt é baixado com o mesmo User-Agent de navegador da coleta — o
    User-Agent padrão do urllib recebe 403 e faria o parser bloquear tudo.
    """
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = requests.get(urljoin(BASE, "/robots.txt"), headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"[robots] http {r.status_code} ao ler robots.txt; abortando.")
            return False
        rp.parse(r.content.decode("utf-8", errors="replace").splitlines())
    except requests.RequestException as exc:
        print(f"[robots] não foi possível ler robots.txt ({exc}); abortando.")
        return False
    return rp.can_fetch(HEADERS["User-Agent"], url)


def baixar(url: str, tentativas: int = 3) -> str | None:
    """Baixa uma página com retries e decodificação UTF-8 explícita."""
    for n in range(1, tentativas + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.content.decode("utf-8", errors="replace")
            print(f"[http {r.status_code}] {url}")
        except requests.RequestException as exc:
            print(f"[erro rede {n}/{tentativas}] {url}: {exc}")
        time.sleep(DELAY_SEGUNDOS * n)
    return None


def _num(texto: str) -> int | None:
    """Extrai o primeiro número inteiro de um texto (ex.: 'R$ 690.000')."""
    digitos = re.sub(r"[^\d]", "", texto or "")
    return int(digitos) if digitos else None


def parse_card(card) -> dict | None:
    """Extrai os campos de interesse de um card de imóvel."""
    a = card.select_one("a.link_rawLink__Tabnf")
    if not a or not a.get("href"):
        return None
    href = a["href"]

    preco_tag = card.select_one("p[aria-label='Preço'] b") or card.select_one(
        ".style_price__Dk6Z9 b"
    )
    preco = _num(preco_tag.get_text()) if preco_tag else None

    enderecos = [p.get_text(strip=True) for p in card.select("address p")]
    rua = enderecos[0] if enderecos else None
    bairro = cidade = uf = None
    if len(enderecos) > 1:
        # formato "Bairro, Cidade/UF"
        m = re.match(r"(.+),\s*(.+)/(\w{2})", enderecos[1])
        if m:
            bairro, cidade, uf = m.group(1), m.group(2), m.group(3)

    # features: "120 área útil", "3 Quartos", "3 Banheiros", "2 Garagens"
    dados = {v: None for v in FEATURE_KEYS.values()}
    area_util = None
    for p in card.select("span.style_list__XnasM p"):
        rotulo = (p.get("aria-label") or "").lower()
        if "área útil" in rotulo or "area util" in rotulo:
            area_util = _num(rotulo)
            continue
        for chave, coluna in FEATURE_KEYS.items():
            if chave in rotulo:
                dados[coluna] = _num(rotulo)

    # tipo e área total a partir da URL: /imovel/casa-a-venda-...-220m2-RS690000/
    tipo = None
    m_tipo = re.search(r"/imovel/([a-z-]+?)-a-venda", href)
    if m_tipo:
        tipo = m_tipo.group(1).replace("-", " ")
    m_area = re.search(r"-(\d+)m2-", href)
    area_total = int(m_area.group(1)) if m_area else None

    return {
        "id": card.get("id", "").replace("rc-", ""),
        "tipo": tipo,
        "preco": preco,
        "area_util": area_util,
        "area_total": area_total,
        "quartos": dados["quartos"],
        "banheiros": dados["banheiros"],
        "vagas": dados["vagas"],
        "salas": dados["salas"],
        "rua": rua,
        "bairro": bairro,
        "cidade": cidade,
        "uf": uf,
        "url": urljoin(BASE, href),
    }


def scrape_cidade(uf: str, cidade: str, max_paginas: int) -> list[dict]:
    base_url = LISTA_URL.format(uf=uf.lower(), cidade=cidade.lower())
    if not pode_coletar(base_url):
        raise SystemExit(f"[robots] coleta não permitida para {base_url}")

    vistos: set[str] = set()
    resultados: list[dict] = []
    for pg in range(1, max_paginas + 1):
        url = base_url if pg == 1 else f"{base_url}?pg={pg}"
        html = baixar(url)
        if not html:
            break
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select('div[id^="rc-"]')
        if not cards:
            print(f"[fim] página {pg} sem cards; encerrando.")
            break
        novos = 0
        for card in cards:
            item = parse_card(card)
            if item and item["id"] and item["id"] not in vistos:
                vistos.add(item["id"])
                resultados.append(item)
                novos += 1
        print(f"[pg {pg}] {novos} novos imóveis (total {len(resultados)})")
        if novos == 0:  # sem itens inéditos => provavelmente repetindo
            break
        time.sleep(DELAY_SEGUNDOS)
    return resultados


def salvar_csv(linhas: list[dict], cidade: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = RAW_DIR / f"imoveis_{cidade.lower()}_{date.today():%Y%m%d}.csv"
    if not linhas:
        print("[aviso] nenhuma linha para salvar.")
        return arquivo
    with arquivo.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"[ok] {len(linhas)} imóveis salvos em {arquivo}")
    return arquivo


def main() -> None:
    ap = argparse.ArgumentParser(description="Scraper de imóveis do Chaves na Mão")
    ap.add_argument("--uf", default="to", help="UF, ex.: to, sp, rj")
    ap.add_argument("--cidade", default="palmas", help="Cidade, ex.: palmas")
    ap.add_argument("--max-paginas", type=int, default=30)
    args = ap.parse_args()

    linhas = scrape_cidade(args.uf, args.cidade, args.max_paginas)
    salvar_csv(linhas, args.cidade)


if __name__ == "__main__":
    main()

from __future__ import annotations
from bs4 import BeautifulSoup
import re
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    return re.sub(r"\s+", " ", s).strip()


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)
    h1 = soup.find('h1')
    return clean_text(h1.get_text(separator=' ')) if h1 else ''


def extract_main_text(soup: BeautifulSoup) -> str:
    for selector in ['main', 'article']:
        node = soup.find(selector)
        if node:
            for bad in node.find_all(['script','style','noscript','svg','nav','footer','header','aside']):
                bad.decompose()
            txt = clean_text(node.get_text(separator=' '))
            if len(txt.split()) > 50:
                return txt
    node = soup.find(attrs={'role': 'main'})
    if node:
        for bad in node.find_all(['script','style','noscript','svg','nav','footer','header','aside']):
            bad.decompose()
        txt = clean_text(node.get_text(separator=' '))
        if len(txt.split()) > 50:
            return txt
    for bad in soup.find_all(['script','style','noscript','svg']):
        bad.decompose()
    ps = [p.get_text(separator=' ') for p in soup.find_all('p')]
    txt = clean_text(' '.join(ps)) if ps else clean_text(soup.get_text(separator=' '))
    return txt


def parse_html(html: str) -> tuple[str, str]:
    try:
        soup = BeautifulSoup(html, 'lxml')
        return extract_title(soup), extract_main_text(soup)
    except Exception:
        return '', ''


def sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r'[.!?]+', text)
    return len([p for p in parts if p.strip()])


def fetch_url(url: str, timeout: int = 15) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ''

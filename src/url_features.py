"""Offline lexical features for a URL, without visiting the website."""
from __future__ import annotations

from collections import Counter
import ipaddress
import math
import re
from urllib.parse import urlsplit

import pandas as pd

FEATURE_COLUMNS = [
    "url_length", "host_length", "path_length", "query_length",
    "dot_count", "hyphen_count", "underscore_count", "at_count",
    "question_count", "ampersand_count", "equals_count", "percent_count",
    "slash_count", "digit_count", "letter_count", "special_count",
    "digit_ratio", "url_entropy", "is_https", "has_ip_host",
    "subdomain_count", "hostname_token_count", "has_punycode",
    "has_port", "has_login_token", "has_verify_token",
    "has_account_token", "has_secure_token", "has_download_token",
]


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values())


def extract_url_features(url: str) -> dict[str, float]:
    """Extract numeric features from a URL string. Never performs network calls."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    text = url.strip()
    parseable = text if "://" in text else "http://" + text
    parsed = urlsplit(parseable)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    digits = sum(c.isdigit() for c in text)
    letters = sum(c.isalpha() for c in text)
    try:
        ipaddress.ip_address(host)
        ip_host = 1
    except ValueError:
        ip_host = 0
    # A public-suffix-aware subdomain count is a future improvement.
    parts = [piece for piece in host.split(".") if piece]
    subdomains = max(0, len(parts) - 2)
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 1
    lower = text.lower()
    tokens = re.split(r"[^a-z0-9]+", lower)
    tokens = set(tokens)
    return {
        "url_length": len(text),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": text.count("."),
        "hyphen_count": text.count("-"),
        "underscore_count": text.count("_"),
        "at_count": text.count("@"),
        "question_count": text.count("?"),
        "ampersand_count": text.count("&"),
        "equals_count": text.count("="),
        "percent_count": text.count("%"),
        "slash_count": text.count("/"),
        "digit_count": digits,
        "letter_count": letters,
        "special_count": len(text) - digits - letters,
        "digit_ratio": digits / max(len(text), 1),
        "url_entropy": _entropy(text),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_ip_host": ip_host,
        "subdomain_count": subdomains,
        "hostname_token_count": len(re.findall(r"[a-z0-9]+", host)),
        "has_punycode": int("xn--" in host),
        "has_port": has_port,
        "has_login_token": int(any(t in tokens for t in ("login", "signin", "logon"))),
        "has_verify_token": int(any(t in tokens for t in ("verify", "verification", "validate"))),
        "has_account_token": int("account" in tokens),
        "has_secure_token": int("secure" in tokens),
        "has_download_token": int("download" in tokens),
    }


def make_feature_frame(urls: list[str] | pd.Series) -> pd.DataFrame:
    """Return features in the same fixed order used during training."""
    return pd.DataFrame([extract_url_features(url) for url in urls], columns=FEATURE_COLUMNS).astype(float)

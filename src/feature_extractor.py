"""
URL feature extraction module for Sri Lankan phishing and smishing detection.
Extracts 22 lexical, host structure, and localized (.lk / brand) features from a URL.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Any

# Official Sri Lankan Brand Dictionary: brand_token -> list of official legitimate root domains
SL_OFFICIAL_BRANDS: Dict[str, List[str]] = {
    "boc": ["boc.lk", "bankofceylon.lk"],
    "sampath": ["sampath.lk", "sampathvishwa.com"],
    "vishwa": ["sampathvishwa.com", "sampath.lk"],
    "combank": ["combank.lk", "combankdigital.com"],
    "hnb": ["hnb.net", "hnb.lk"],
    "peoplesbank": ["peoplesbank.lk"],
    "nsb": ["nsb.lk"],
    "seylan": ["seylan.lk"],
    "frimi": ["frimi.lk", "ntb.lk"],
    "ndb": ["ndbbank.com"],
    "dfcc": ["dfcc.lk"],
    "cargillsbank": ["cargillsbank.com"],
    "panasia": ["pabcbank.com"],
    "slpost": ["slpost.gov.lk"],
    "srilankapost": ["slpost.gov.lk"],
    "ceb": ["ceb.lk"],
    "leco": ["leco.lk"],
    "waterboard": ["waterboard.lk"],
    "dialog": ["dialog.lk", "wow.lk"],
    "mobitel": ["mobitel.lk", "slt.lk", "sltmobitel.lk"],
    "sltmobitel": ["slt.lk", "sltmobitel.lk", "mobitel.lk"],
    "hutch": ["hutch.lk"],
    "airtel": ["airtel.lk"],
    "ird": ["ird.gov.lk"],
    "govlk": ["gov.lk"],
    "epf": ["epf.lk", "cbsl.gov.lk"],
    "cbsl": ["cbsl.gov.lk"],
    "lankaqr": ["lankapay.net", "cbsl.gov.lk"],
    "lankapay": ["lankapay.net"],
    "daraz": ["daraz.lk"],
    "kapruka": ["kapruka.com"],
    "ikman": ["ikman.lk"],
    "pickme": ["pickme.lk"],
    "helakuru": ["helakuru.lk", "bhasha.lk"],
    "srilankan": ["srilankan.com"],
}

# List of all canonical Sri Lankan brand domains for Levenshtein distance calculation
CANONICAL_SL_DOMAINS: List[str] = [
    "boc.lk", "sampath.lk", "sampathvishwa.com", "combank.lk", "hnb.lk",
    "peoplesbank.lk", "nsb.lk", "seylan.lk", "frimi.lk", "slpost.gov.lk",
    "ceb.lk", "leco.lk", "dialog.lk", "mobitel.lk", "slt.lk", "hutch.lk",
    "ird.gov.lk", "gov.lk", "cbsl.gov.lk", "lankapay.net", "daraz.lk",
    "ikman.lk", "pickme.lk", "waterboard.lk"
]

# High-risk TLDs frequently abused in SLCERT / FinCSIRT phishing & smishing campaigns
HIGH_RISK_TLDS = {
    "top", "xyz", "icu", "vip", "buzz", "club", "online", "site", "live",
    "cn", "ru", "tk", "ml", "ga", "cf", "gq", "work", "click", "link",
    "rest", "fit", " loan", "win", "bid", "stream", "pw", "cc", "shop", "sbs", "cyou"
}

# Known URL shortening services used in SMS smishing
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "rb.gy", "cutt.ly",
    "shorturl.at", "tiny.cc", "ow.ly", "rebrand.ly", "v.gd", "lnkd.in"
}

# Social-engineering urgency / security action tokens
URGENCY_KEYWORDS = [
    "verify", "verification", "suspend", "suspended", "customs", "parcel",
    "clearance", "unlock", "otp", "kyc", "subsidy", "refund", "confirm",
    "login", "signin", "update", "secure", "account", "urgent", "blocked",
    "tax", "arrears", "disconnection", "reward", "claim", "validate", "password"
]

# Ordered list of the 22 model features
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "url_entropy",
    "digit_ratio",
    "special_char_count",
    "num_hyphens_in_domain",
    "has_at_or_double_slash",
    "uses_https",
    "https_token_in_domain",
    "has_ip_address",
    "num_subdomains",
    "is_shortened_url",
    "has_suspicious_ext",
    "query_param_count",
    "is_official_lk_domain",
    "fake_lk_in_url",
    "sl_brand_mentioned",
    "sl_brand_domain_mismatch",
    "min_brand_edit_distance",
    "high_risk_tld",
    "urgency_keyword_count",
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "url_length": "Total character length of the full URL",
    "hostname_length": "Character length of the hostname/domain portion",
    "path_length": "Character length of the URL path and endpoint",
    "url_entropy": "Shannon information entropy of the URL string (measures randomness)",
    "digit_ratio": "Proportion of numeric digits relative to total URL length",
    "special_char_count": "Total count of special symbols (-, _, @, ?, =, %, &) in the URL",
    "num_hyphens_in_domain": "Count of hyphen (-) characters inside the hostname",
    "has_at_or_double_slash": "Binary flag for @ credential trick or // open-redirect in URL path",
    "uses_https": "Binary flag indicating whether the URL uses HTTPS protocol (1=HTTPS, 0=HTTP)",
    "https_token_in_domain": "Deceptive presence of 'https' or 'ssl' inside the domain name itself",
    "has_ip_address": "Binary flag indicating if the host is a raw IPv4/hex address instead of a domain",
    "num_subdomains": "Subdomain depth level (number of subdomain prefixes before root domain)",
    "is_shortened_url": "Binary flag for SMS URL shorteners (bit.ly, tinyurl, cutt.ly, etc.)",
    "has_suspicious_ext": "Binary flag for risky path extensions (.php, .apk, .exe, .scr, .zip)",
    "query_param_count": "Number of query parameters passed in the URL string",
    "is_official_lk_domain": "Verified Sri Lankan (.lk, .gov.lk, .ac.lk, .com.lk, .org.lk) root domain",
    "fake_lk_in_url": "Deceptive 'lk', 'gov-lk', or 'srilanka' token hosted on a foreign non-.lk TLD",
    "sl_brand_mentioned": "Presence of a major Sri Lankan bank, telco, utility, or gov brand keyword",
    "sl_brand_domain_mismatch": "SL brand keyword appears in URL, but root domain is NOT the official brand domain",
    "min_brand_edit_distance": "Normalized Levenshtein similarity distance to closest SL brand (detects typosquatting)",
    "high_risk_tld": "Domain uses a high-risk TLD (.top, .xyz, .icu, .vip, .buzz, .online, .sbs)",
    "urgency_keyword_count": "Count of social-engineering urgency words (verify, customs, parcel, otp, kyc, suspend)",
}


def compute_shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy (bits per character) of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def get_effective_root_domain(hostname: str) -> str:
    """
    Extract the effective registered root domain (e.g. 'boc.lk', 'slpost.gov.lk', 'example.com').
    Handles Sri Lankan second-level domains (.gov.lk, .ac.lk, .com.lk, .org.lk, .sch.lk, .net.lk).
    """
    hostname = hostname.lower().strip(".")
    if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname):
        return hostname
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    sl_second_levels = {"gov", "ac", "com", "org", "sch", "net", "edu", "int", "mil"}
    if parts[-1] == "lk" and parts[-2] in sl_second_levels and len(parts) >= 3:
        return ".".join(parts[-3:])
    if parts[-1] in {"uk", "au", "in", "sg"} and parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def extract_url_features(url: str) -> Dict[str, float]:
    """
    Extract all 22 structured cybersecurity & Sri Lankan context features from a raw URL.
    """
    raw_url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        parse_target = "http://" + raw_url
        uses_https = 0
    else:
        parse_target = raw_url
        uses_https = 1 if raw_url.lower().startswith("https://") else 0

    parsed = urlparse(parse_target)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_lower = raw_url.lower()

    # 1-3. Length metrics
    url_length = len(raw_url)
    hostname_length = len(hostname)
    path_length = len(path)

    # 4. Entropy
    url_entropy = round(compute_shannon_entropy(raw_url), 4)

    # 5. Digit ratio
    num_digits = sum(c.isdigit() for c in raw_url)
    digit_ratio = round(num_digits / max(url_length, 1), 4)

    # 6. Special character count
    special_chars = set("-_@?=%.&~+")
    special_char_count = sum(c in special_chars for c in raw_url)

    # 7. Hyphens in domain
    num_hyphens_in_domain = hostname.count("-")

    # 8. @ symbol or // in path/query
    after_scheme = re.sub(r"^[a-zA-Z0-9+.-]+://", "", raw_url)
    has_at_or_double_slash = 1 if ("@" in raw_url or "//" in after_scheme) else 0

    # 9. uses_https already determined

    # 10. Deceptive 'https' or 'ssl' inside hostname
    https_token_in_domain = 1 if ("https" in hostname or "ssl" in hostname) else 0

    # 11. Raw IP address host
    has_ip_address = 1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname) or re.match(r"^0x[0-9a-f]+", hostname) else 0

    # 12. Number of subdomains
    root_domain = get_effective_root_domain(hostname)
    if has_ip_address or not hostname:
        num_subdomains = 0
    else:
        prefix = hostname[: max(0, len(hostname) - len(root_domain))].strip(".")
        if not prefix:
            num_subdomains = 0
        else:
            sub_parts = [p for p in prefix.split(".") if p and p != "www"]
            num_subdomains = len(sub_parts)

    # 13. Shortened URL service
    is_shortened_url = 1 if (root_domain in SHORTENING_SERVICES or hostname in SHORTENING_SERVICES) else 0

    # 14. Suspicious file extension in path
    path_lower = path.lower()
    suspicious_exts = (".php", ".exe", ".apk", ".scr", ".zip", ".rar", ".bat", ".sh", ".cgi", ".bin")
    has_suspicious_ext = 1 if any(path_lower.endswith(ext) or (ext + "?") in full_lower for ext in suspicious_exts) else 0

    # 15. Query parameter count
    query_params = parse_qs(query, keep_blank_values=True)
    query_param_count = len(query_params)

    # 16. Is official .lk domain
    is_official_lk_domain = 1 if (not has_ip_address and hostname.endswith(".lk")) else 0

    # 17. Fake 'lk' or 'srilanka' in subdomain/domain/path while NOT on an official .lk domain
    lk_pattern = re.compile(r"(^|[-./_])(lk|gov-lk|com-lk|ac-lk|srilanka|lanka|ceylon)([-./_]|$)")
    fake_lk_in_url = 1 if (is_official_lk_domain == 0 and bool(lk_pattern.search(full_lower))) else 0

    # 18 & 19. Sri Lankan brand mention & domain mismatch
    sl_brand_mentioned = 0
    sl_brand_domain_mismatch = 0
    for brand_token, official_domains in SL_OFFICIAL_BRANDS.items():
        if brand_token in full_lower:
            sl_brand_mentioned = 1
            if root_domain not in official_domains and not is_official_lk_domain:
                sl_brand_domain_mismatch = 1
                break
            elif root_domain not in official_domains and is_official_lk_domain:
                # Even on .lk, check if it's an unrelated domain impersonating a bank
                if not any(hostname.endswith(d) for d in official_domains):
                    sl_brand_domain_mismatch = 1
                    break

    # 20. Minimum normalized Levenshtein edit distance to SL brand root domains (detects typosquatting)
    if not hostname or has_ip_address:
        min_brand_edit_distance = 1.0
    else:
        host_core = root_domain.split(".")[0]
        # Also inspect subdomain tokens for brand lookalikes
        tokens = [t for t in re.split(r"[.-]", hostname) if len(t) >= 3 and t not in {"www", "com", "net", "org", "gov", "top", "xyz", "online", "site"}]
        if not tokens:
            tokens = [host_core]
        best_norm_dist = 1.0
        for tok in tokens:
            for canon in CANONICAL_SL_DOMAINS:
                canon_core = canon.split(".")[0]
                dist = levenshtein_distance(tok, canon_core)
                norm_dist = dist / max(len(tok), len(canon_core), 1)
                # If exact match on official domain, distance is 0.0 (safe); if typosquatted (0 < norm_dist <= 0.35), flag closeness!
                if norm_dist < best_norm_dist:
                    best_norm_dist = norm_dist
        min_brand_edit_distance = round(best_norm_dist, 4)

    # 21. High-risk TLD
    tld = hostname.split(".")[-1] if "." in hostname and not has_ip_address else ""
    high_risk_tld = 1 if tld in HIGH_RISK_TLDS else 0

    # 22. Urgency / social-engineering security keyword count
    urgency_keyword_count = sum(1 for kw in URGENCY_KEYWORDS if kw in full_lower)

    return {
        "url_length": float(url_length),
        "hostname_length": float(hostname_length),
        "path_length": float(path_length),
        "url_entropy": float(url_entropy),
        "digit_ratio": float(digit_ratio),
        "special_char_count": float(special_char_count),
        "num_hyphens_in_domain": float(num_hyphens_in_domain),
        "has_at_or_double_slash": float(has_at_or_double_slash),
        "uses_https": float(uses_https),
        "https_token_in_domain": float(https_token_in_domain),
        "has_ip_address": float(has_ip_address),
        "num_subdomains": float(num_subdomains),
        "is_shortened_url": float(is_shortened_url),
        "has_suspicious_ext": float(has_suspicious_ext),
        "query_param_count": float(query_param_count),
        "is_official_lk_domain": float(is_official_lk_domain),
        "fake_lk_in_url": float(fake_lk_in_url),
        "sl_brand_mentioned": float(sl_brand_mentioned),
        "sl_brand_domain_mismatch": float(sl_brand_domain_mismatch),
        "min_brand_edit_distance": float(min_brand_edit_distance),
        "high_risk_tld": float(high_risk_tld),
        "urgency_keyword_count": float(urgency_keyword_count),
    }

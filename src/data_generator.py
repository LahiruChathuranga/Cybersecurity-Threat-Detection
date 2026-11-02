"""
Dataset generation script for Sri Lankan phishing and legitimate URLs.
Builds a balanced dataset of 6,000 URLs across 11 categories and extracts the 22 model features.
"""

import os
import random
import string
import numpy as np
import pandas as pd
from src.feature_extractor import extract_url_features, FEATURE_NAMES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _rand_token(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _rand_digits(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def generate_legitimate_sl_urls(n: int = 1800) -> list[dict]:
    """Generate authentic Sri Lankan legitimate URLs across banking, gov, education, telco, and commerce."""
    sl_domains = [
        ("www.boc.lk", "Banking"),
        ("online.boc.lk", "Banking"),
        ("www.sampath.lk", "Banking"),
        ("www.sampathvishwa.com", "Banking"),
        ("www.combank.lk", "Banking"),
        ("www.combankdigital.com", "Banking"),
        ("www.hnb.net", "Banking"),
        ("www.peoplesbank.lk", "Banking"),
        ("www.nsb.lk", "Banking"),
        ("www.seylan.lk", "Banking"),
        ("www.frimi.lk", "Banking"),
        ("www.ndbbank.com", "Banking"),
        ("www.dfcc.lk", "Banking"),
        ("www.lankapay.net", "Banking"),
        ("www.gov.lk", "E-Government"),
        ("www.ird.gov.lk", "E-Government"),
        ("www.slpost.gov.lk", "E-Government"),
        ("www.cbsl.gov.lk", "E-Government"),
        ("www.epf.lk", "E-Government"),
        ("www.immigration.gov.lk", "E-Government"),
        ("www.dmt.gov.lk", "E-Government"),
        ("www.customs.gov.lk", "E-Government"),
        ("www.elections.gov.lk", "E-Government"),
        ("www.police.lk", "E-Government"),
        ("ceb.lk", "Utilities"),
        ("www.ceb.lk", "Utilities"),
        ("www.leco.lk", "Utilities"),
        ("www.waterboard.lk", "Utilities"),
        ("www.dialog.lk", "Telco"),
        ("www.mobitel.lk", "Telco"),
        ("www.slt.lk", "Telco"),
        ("www.hutch.lk", "Telco"),
        ("www.airtel.lk", "Telco"),
        ("www.mrt.ac.lk", "Academic"),
        ("www.cmb.ac.lk", "Academic"),
        ("www.pdn.ac.lk", "Academic"),
        ("www.sjp.ac.lk", "Academic"),
        ("www.kln.ac.lk", "Academic"),
        ("www.sliit.lk", "Academic"),
        ("www.nsbm.ac.lk", "Academic"),
        ("www.daraz.lk", "E-Commerce"),
        ("www.ikman.lk", "E-Commerce"),
        ("www.pickme.lk", "E-Commerce"),
        ("www.kapruka.com", "E-Commerce"),
        ("www.keellssuper.com", "E-Commerce"),
        ("www.cargillsfoodcity.com", "E-Commerce"),
        ("www.srilankan.com", "Travel"),
        ("www.adaderana.lk", "Media"),
        ("www.dailymirror.lk", "Media"),
        ("www.sundaytimes.lk", "Media"),
    ]

    normal_paths = [
        "/personal-banking/accounts",
        "/corporate/treasury-rates",
        "/about-us/board-of-directors",
        "/services/online-bill-payment",
        "/tariffs-and-charges",
        "/contact-us/branch-network",
        "/downloads/annual-report-2025.pdf",
        "/en/citizen-services/passport-renewal",
        "/sin/news/press-release",
        "/tam/notices/general-information",
        "/academic/postgraduate-studies/msc-cs",
        "/customer-support/faq",
        "/products/home-broadband/fiber",
        "/promotions/credit-card-offers",
        "/ SVRClientWeb/ActionController",
        "/portal/home",
        "/ebilling/view-statement",
        "/track-trace/domestic-ems",
        "/marketplace/electronics/mobile-phones",
        "/careers/vacancies",
    ]

    # Include some realistic "hard benign" paths on official SL websites (e.g. login/verify/account pages, HTTP legacy gov pages, deep subdomains)
    hard_benign_paths = [
        "/login/verify-session.php?session_id={tok}&lang=en&redirect=account",
        "/account/update-kyc-details?ref={num}&step=confirm",
        "/customs/parcel-clearance-guidelines?doc={num}&dept=slpost&type=tax",
        "/auth/signin?client_id={tok}&state={tok}&response_type=code",
        "/online-banking/password-reset/confirm?token={tok}-{tok}",
        "/ebill/disconnection-notice-policy?acct={num}&region=colombo-03",
    ]

    records = []
    for i in range(n):
        domain, subcat = random.choice(sl_domains)
        # 14% of legitimate SL government/legacy portals still use HTTP or have complex security-like paths
        if random.random() < 0.14:
            scheme = "http" if ("gov.lk" in domain or random.random() < 0.45) else "https"
            # Sometimes add a deeper legitimate subdomain
            if random.random() < 0.35:
                sub_prefix = random.choice(["portal.auth", "eservices.sec", "ebank.my", "lms.student", "pay.online"])
                host = f"{sub_prefix}.{domain.replace('www.', '')}"
            else:
                host = domain
            path = random.choice(hard_benign_paths).format(tok=_rand_token(10), num=_rand_digits(7))
        else:
            scheme = "https" if random.random() < 0.92 else "http"
            host = domain
            path = random.choice(normal_paths).strip()
            if random.random() < 0.35:
                path += f"?id={_rand_digits(4)}&lang={random.choice(['en', 'si', 'ta'])}"
            elif random.random() < 0.15:
                path += f"/{_rand_token(6)}"

        url = f"{scheme}://{host}{path}"
        records.append({
            "url": url,
            "category": f"SL_Legitimate_{subcat}",
            "label": 0
        })
    return records


def generate_legitimate_global_urls(n: int = 1200) -> list[dict]:
    """Generate global legitimate URLs including developer, cloud, education, and hard benign links."""
    global_domains = [
        "www.google.com", "mail.google.com", "drive.google.com", "accounts.google.com",
        "github.com", "gist.github.com", "stackoverflow.com", "en.wikipedia.org",
        "www.microsoft.com", "login.microsoftonline.com", "portal.azure.com",
        "aws.amazon.com", "console.aws.amazon.com", "www.coursera.org", "www.edx.org",
        "www.linkedin.com", "www.youtube.com", "zoom.us", "slack.com", "www.paypal.com",
        "www.apple.com", "developer.apple.com", "pypi.org", "anaconda.org", "huggingface.co",
        "www.bbc.com", "www.reuters.com", "www.Booking.com", "www.agoda.com", "www.aliexpress.com"
    ]

    paths = [
        "/search?q=sri+lanka+machine+learning",
        "/orgs/community/discussions/{num}",
        "/questions/{num}/how-to-interpret-shap-values",
        "/wiki/Sri_Lanka_Telecommunications",
        "/en-us/security/business/identity-access",
        "/learn/machine-learning-specialization",
        "/watch?v={tok}&ab_channel=TechLK",
        "/j/{num}?pwd={tok}",
        "/project/scikit-learn/1.9.1/",
        "/docs/stable/modules/svm.html",
        "/hotels/sri-lanka/colombo-deals.html?checkIn=2026-12-01&rooms=1",
    ]

    # Hard benign global URLs (multi-hyphen legitimate domains, legitimate bit.ly links, or legitimate .xyz/.online tech sites)
    hard_global_hosts = [
        "my-cloud-service-portal.com",
        "sri-lanka-tourism-alliance.org",
        "colombo-software-engineering-meetup.net",
        "open-source-cyber-defense-hub.io",
        "bit.ly",
        "tinyurl.com",
        "web3-developer-docs.xyz",
        "lk-expat-community-forum.com",
        "ceylon-tea-exporters-directory.com",
        "south-asia-fintech-summit-2026.org",
    ]

    records = []
    for i in range(n):
        if random.random() < 0.22:
            # Hard benign global case
            host = random.choice(hard_global_hosts)
            scheme = "https" if random.random() < 0.78 else "http"
            if host in {"bit.ly", "tinyurl.com"}:
                path = f"/{_rand_token(7)}"
            else:
                path = random.choice([
                    f"/account/login-verify?token={_rand_token(12)}&next=/dashboard",
                    f"/articles/srilanka-boc-sampath-fintech-review-{_rand_digits(4)}",
                    f"/customs-tariff-calculator/import-duty.php?hs={_rand_digits(6)}&currency=LKR",
                    f"/events/register?id={_rand_digits(5)}&utm_source=newsletter&utm_medium=email",
                ])
        else:
            host = random.choice(global_domains).lower()
            scheme = "https" if random.random() < 0.94 else "http"
            path = random.choice(paths).format(num=_rand_digits(7), tok=_rand_token(11))

        url = f"{scheme}://{host}{path}"
        records.append({
            "url": url,
            "category": "Global_Legitimate",
            "label": 0
        })
    return records


def generate_sl_phishing_urls(n: int = 2100) -> list[dict]:
    """
    Generate localized Sri Lankan phishing & smishing URLs modeled after SLCERT & FinCSIRT advisories:
    - SL Post customs parcel clearance fee scams
    - Sampath Vishwa / BOC / ComBank / HNB / FriMi OTP & KYC suspension lures
    - CEB / LECO electricity bill disconnection scams
    - Gov.lk / IRD tax refund or welfare ("Aswesuma") cash grant phishing
    - Dialog / Mobitel mega data reward or SIM block lures
    - Typosquatted SL domains & IP-hosted credential harvesters
    """
    records = []
    phishing_tlds = ["top", "xyz", "icu", "vip", "buzz", "online", "site", "sbs", "cyou", "shop", "club", "live", "cn", "ru", "com", "net", "org", "info"]

    attack_templates = [
        # 1. SL Post Customs & Parcel Smishing (Huge SLCERT wave)
        lambda: (
            f"{random.choice(['http', 'https'])}://slpost-customs-{random.choice(['clearance', 'parcel', 'delivery', 'tax'])}-lk.{random.choice(phishing_tlds)}"
            f"/track/confirm-address.php?parcel_id=LK{_rand_digits(8)}&fee=LKR450",
            "SL_Phishing_Postal_Scam"
        ),
        lambda: (
            f"https://srilankapost.gov.lk.{random.choice(['parcel-hold', 'ems-verify', 'customs-pay'])}.{random.choice(phishing_tlds)}"
            f"/clearance/payment?id={_rand_token(8)}",
            "SL_Phishing_Postal_Scam"
        ),
        # 2. Banking Credential Harvesting (Sampath Vishwa, BOC, ComBank, HNB, PeoplesBank, FriMi)
        lambda: (
            f"{random.choice(['http', 'https'])}://{random.choice(['sampathvishwa', 'boc-online', 'combank-digital', 'hnb-online', 'peoplesbank-lk', 'frimi-wallet'])}"
            f"-{random.choice(['verify', 'security', 'kyc-update', 'unlock', 'auth'])}.{random.choice(phishing_tlds)}"
            f"/login/validate-otp.php?session={_rand_token(14)}",
            "SL_Phishing_Banking"
        ),
        lambda: (
            f"https://{random.choice(['boc.lk', 'sampath.lk', 'combank.lk', 'hnb.lk', 'nsb.lk'])}"
            f".{random.choice(['account-verify', 'secure-login', 'ssl-update', 'ibanking-lk'])}.{random.choice(phishing_tlds)}"
            f"/SVRClientWeb/login?user_auth={_rand_digits(6)}",
            "SL_Phishing_Banking"
        ),
        # 3. Typosquatting SL Banks (subtle domain misspellings on .com / .net / .lk lookalikes)
        lambda: (
            f"https://www.{random.choice(['smapathvishwa', 'sampath-vishwa', 'combnak-lk', 'b0c-online', 'peoplesbnak', 'hnb-ibanking', 'seylan-bank-lk'])}"
            f".{random.choice(['com', 'net', 'org', 'top', 'online'])}/ibanking/{random.choice(['signin', 'verify', 'auth', 'portal'])}?id={_rand_digits(5)}",
            "SL_Phishing_Typosquatting"
        ),
        # 4. CEB / LECO / Waterboard Utility Disconnection Scams
        lambda: (
            f"{random.choice(['http', 'https'])}://{random.choice(['ceb-lk', 'leco-bill', 'waterboard-lk'])}-"
            f"{random.choice(['disconnection-notice', 'arrears-pay', 'subsidy-claim', 'urgent-bill'])}.{random.choice(phishing_tlds)}"
            f"/pay/instant-settlement?acct={_rand_digits(10)}",
            "SL_Phishing_Utility_Scam"
        ),
        # 5. Gov.lk / IRD Tax / EPF / Aswesuma Welfare Lures
        lambda: (
            f"{random.choice(['http', 'https'])}://gov-lk-{random.choice(['aswesuma-grant', 'ird-tax-refund', 'epf-claim', 'presidential-fund', 'fuel-pass'])}"
            f".{random.choice(phishing_tlds)}/citizen/verify-nic.php?nic={_rand_digits(9)}V",
            "SL_Phishing_EGov_Scam"
        ),
        # 6. Telco (Dialog / SLT-Mobitel / Hutch) Free Data & SIM KYC Suspension
        lambda: (
            f"{random.choice(['http', 'https'])}://{random.choice(['dialog-lk', 'sltmobitel', 'mobitel-lk', 'hutch-lk'])}-"
            f"{random.choice(['50gb-reward', 'sim-kyc-suspend', 'mega-reload', 'loyalty-claim'])}.{random.choice(phishing_tlds)}"
            f"/claim/activate?msisdn=07{_rand_digits(8)}",
            "SL_Phishing_Telco_Scam"
        ),
        # 7. Raw IP Address or @ Redirect Phishing targeting Sri Lankan users
        lambda: (
            f"http://{random.choice(['103.145.22', '185.220.101', '45.133.19', '192.241.210'])}.{random.randint(2, 254)}"
            f"/{random.choice(['boc-lk', 'sampath-vishwa', 'slpost-ems', 'combank-auth', 'gov-lk'])}/login.php?token={_rand_token(10)}",
            "SL_Phishing_IP_or_Redirect"
        ),
        lambda: (
            f"https://www.{random.choice(['boc.lk', 'sampath.lk', 'gov.lk', 'slpost.gov.lk'])}@"
            f"{random.choice(['auth-server-lk.xyz', 'secure-gateway-24.top', 'verify-client.icu'])}/login/confirm",
            "SL_Phishing_IP_or_Redirect"
        ),
    ]

    for _ in range(n):
        builder = random.choice(attack_templates)
        url, cat = builder()
        records.append({
            "url": url,
            "category": cat,
            "label": 1
        })
    return records


def generate_stealth_and_global_phishing_urls(n: int = 900) -> list[dict]:
    """
    Generate stealth phishing URLs (including hard phishing cases that mimic clean HTTPS websites,
    shortened smishing URLs, or subtle global credential harvesters without obvious flags).
    """
    records = []

    stealth_templates = [
        # Hard phishing: clean HTTPS .com/.org domain, no hyphens or only 1 hyphen, subtle path
        lambda: (
            f"https://{random.choice(['lankapay-gateway', 'ceylon-digital-auth', 'vishwa-portal', 'colombo-interbank', 'sl-customs-portal', 'lk-citizen-id'])}"
            f".{random.choice(['com', 'net', 'org'])}/{random.choice(['portal', 'client', 'app', 'v2'])}/{_rand_token(6)}"
        ),
        # Hard phishing: shortened SMS smishing link or subtle subdomain
        lambda: (
            f"https://{random.choice(['bit.ly', 'tinyurl.com', 'cutt.ly', 'rb.gy', 'is.gd'])}/{_rand_token(7)}"
            f"?ref={random.choice(['lk', 'post', 'bank', 'auth', 'id'])}"
        ),
        # Subtle typosquatting with clean path (very few special chars, HTTPS enabled)
        lambda: (
            f"https://www.{random.choice(['sampathviswa', 'combankdigtal', 'peoplesbannk', 'srilankap0st', 'diaIog-lk', 'daraz-srilanka', 'pickme-ride'])}"
            f".{random.choice(['com', 'net', 'co'])}/home/index.html"
        ),
        # Global cloud/email credential phishing targeting SL enterprises
        lambda: (
            f"https://microsoft-365-{random.choice(['workspace', 'sharepoint', 'onedrive'])}.{random.choice(['com-portal.net', 'auth-sso.org', 'cloud-lk.xyz'])}"
            f"/owa/auth/logon.aspx?replaceCurrent=1&url={_rand_token(12)}"
        ),
        # APK / malware dropper disguised as SL banking / gov app update
        lambda: (
            f"https://dl.{random.choice(['boc-smartpass', 'vishwa-mobile', 'govpay-srilanka', 'helakuru-pay'])}.{random.choice(['com', 'net', 'top', 'site'])}"
            f"/releases/v4.2/update_{_rand_digits(3)}.apk"
        ),
    ]

    for _ in range(n):
        url = random.choice(stealth_templates)()
        records.append({
            "url": url,
            "category": "Stealth_or_Global_Phishing",
            "label": 1
        })
    return records


def build_dataset(output_dir: str = "data") -> pd.DataFrame:
    """Build the full Sri Lankan Cybersecurity Phishing Dataset and save raw & processed splits."""
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    legit_sl = generate_legitimate_sl_urls(1800)
    legit_global = generate_legitimate_global_urls(1200)
    phish_sl = generate_sl_phishing_urls(2100)
    phish_stealth = generate_stealth_and_global_phishing_urls(900)

    all_records = legit_sl + legit_global + phish_sl + phish_stealth
    random.shuffle(all_records)

    rows = []
    for rec in all_records:
        feats = extract_url_features(rec["url"])
        row = {
            "url": rec["url"],
            "category": rec["category"],
            **feats,
            "label": rec["label"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Introduce realistic real-world cybersecurity ambiguity & feature overlap (~8-10% hard cases):
    # 1. Compromised legitimate .lk or corporate domains hosting phishing pages (no high_risk_tld, valid .lk, HTTPS enabled)
    # 2. Newly launched legitimate Sri Lankan promo/event microsites (hosted on .xyz/.online/.com, use hyphens & urgency words like 'confirm'/'register'/'claim')
    # 3. Zero-day spear-phishing URLs with short clean paths and low entropy
    # 4. Complex legitimate e-banking / e-government URLs with long query tokens, high entropy, and multiple subdomains
    rng = np.random.default_rng(SEED)
    n_rows = len(df)

    # A. Compromised .lk / stealth phishing (~9% of phishing URLs look structurally very similar to benign sites)
    compromised_phish = (df["label"] == 1) & (rng.random(n_rows) < 0.095)
    df.loc[compromised_phish, "is_official_lk_domain"] = rng.choice([0.0, 1.0], size=compromised_phish.sum(), p=[0.4, 0.6])
    df.loc[compromised_phish, "high_risk_tld"] = 0.0
    df.loc[compromised_phish, "sl_brand_domain_mismatch"] = 0.0
    df.loc[compromised_phish, "fake_lk_in_url"] = 0.0
    df.loc[compromised_phish, "uses_https"] = 1.0
    df.loc[compromised_phish, "num_hyphens_in_domain"] = rng.choice([0.0, 1.0], size=compromised_phish.sum(), p=[0.7, 0.3])
    df.loc[compromised_phish, "urgency_keyword_count"] = rng.choice([0.0, 1.0], size=compromised_phish.sum(), p=[0.6, 0.4])
    df.loc[compromised_phish, "url_length"] = rng.integers(32, 68, size=compromised_phish.sum()).astype(float)
    df.loc[compromised_phish, "hostname_length"] = rng.integers(11, 22, size=compromised_phish.sum()).astype(float)

    # B. Legitimate marketing / partner / microsite URLs that trigger heuristic false alarms (~8.5% of benign URLs)
    noisy_benign = (df["label"] == 0) & (rng.random(n_rows) < 0.085)
    df.loc[noisy_benign, "is_official_lk_domain"] = 0.0
    df.loc[noisy_benign, "high_risk_tld"] = rng.choice([0.0, 1.0], size=noisy_benign.sum(), p=[0.5, 0.5])
    df.loc[noisy_benign, "sl_brand_mentioned"] = rng.choice([0.0, 1.0], size=noisy_benign.sum(), p=[0.4, 0.6])
    df.loc[noisy_benign, "sl_brand_domain_mismatch"] = rng.choice([0.0, 1.0], size=noisy_benign.sum(), p=[0.55, 0.45])
    df.loc[noisy_benign, "fake_lk_in_url"] = rng.choice([0.0, 1.0], size=noisy_benign.sum(), p=[0.6, 0.4])
    df.loc[noisy_benign, "urgency_keyword_count"] = rng.integers(1, 4, size=noisy_benign.sum()).astype(float)
    df.loc[noisy_benign, "num_hyphens_in_domain"] = rng.integers(1, 4, size=noisy_benign.sum()).astype(float)
    df.loc[noisy_benign, "url_length"] = rng.integers(58, 115, size=noisy_benign.sum()).astype(float)
    df.loc[noisy_benign, "special_char_count"] = rng.integers(5, 13, size=noisy_benign.sum()).astype(float)

    # C. Realistic label/sensor noise in threat feeds (~1.8% mislabeled or ambiguous redirectors)
    ambiguous_flip = rng.random(n_rows) < 0.018
    df.loc[ambiguous_flip, "label"] = 1 - df.loc[ambiguous_flip, "label"]

    # D. Continuous distribution variance across lexical metrics
    df["url_length"] = np.clip(df["url_length"] + rng.normal(0, 10.0, n_rows), 18, 220).round(0)
    df["hostname_length"] = np.clip(df["hostname_length"] + rng.normal(0, 4.5, n_rows), 6, 85).round(0)
    df["path_length"] = np.clip(df["path_length"] + rng.normal(0, 7.0, n_rows), 1, 140).round(0)
    df["url_entropy"] = np.clip(df["url_entropy"] + rng.normal(0, 0.28, n_rows), 2.5, 5.8).round(4)
    df["digit_ratio"] = np.clip(df["digit_ratio"] + rng.normal(0, 0.035, n_rows), 0.0, 0.65).round(4)
    df["min_brand_edit_distance"] = np.clip(df["min_brand_edit_distance"] + rng.normal(0, 0.09, n_rows), 0.0, 1.0).round(4)

    raw_path = os.path.join(output_dir, "raw", "srilanka_phishing_dataset.csv")
    df.to_csv(raw_path, index=False)
    print(f"[+] Saved full dataset ({len(df)} rows, {len(FEATURE_NAMES)} features) to {raw_path}")
    print(df["label"].value_counts().rename({0: "Legitimate (0)", 1: "Phishing (1)"}))
    print("\nBreakdown by Category:")
    print(df.groupby(["category", "label"]).size())
    return df


if __name__ == "__main__":
    build_dataset()

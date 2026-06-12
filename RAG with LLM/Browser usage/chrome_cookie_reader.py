"""
chrome_cookie_reader.py
-----------------------
Reads cookies from a Chrome profile directly via SQLite + Windows DPAPI.
Drop-in replacement for the browser_cookie3-based export_cookies() function.

Requirements (all standard or easy installs):
    pip install pycryptodome pywin32

Usage:
    from chrome_cookie_reader import export_cookies
    auth_state = export_cookies()  # returns {"cookies": [...], "origins": []}
"""

import os
import json
import shutil
import sqlite3
import base64
import tempfile
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

CHROME_USER_DATA = r"C:\Users\faiza\AppData\Local\Google\Chrome\User Data"
PROFILE_NAME     = "Profile 23"

COOKIE_DOMAINS = [
    ".google.com",
    ".youtube.com",
    ".googleapis.com",
    "docs.google.com",
    "accounts.google.com",
]

# ── DPAPI / AES decryption ─────────────────────────────────────────────────────

def _get_encryption_key(user_data_dir: str) -> bytes:
    """
    Extract the AES key Chrome stores inside Local State.
    Chrome 80+ encrypts cookies with AES-256-GCM; the key itself
    is protected with Windows DPAPI.
    """
    local_state_path = os.path.join(user_data_dir, "Local State")
    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.load(f)

    # The key is base64-encoded and prefixed with "DPAPI"
    encrypted_key_b64 = local_state["os_crypt"]["encrypted_key"]
    encrypted_key     = base64.b64decode(encrypted_key_b64)
    encrypted_key     = encrypted_key[5:]          # strip the "DPAPI" prefix

    # Decrypt with Windows DPAPI
    import win32crypt
    key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
    return key


def _decrypt_cookie_value(encrypted_value: bytes, key: bytes) -> str:
    """
    Decrypt an AES-256-GCM encrypted cookie value.
    Format: b'v10' + 12-byte nonce + ciphertext + 16-byte tag
    Falls back to legacy DPAPI decryption for older Chrome profiles.
    """
    if not encrypted_value:
        return ""

    try:
        # Chrome 80+ format: v10 / v11 prefix + AES-GCM
        if encrypted_value[:3] in (b"v10", b"v11"):
            from Crypto.Cipher import AES
            nonce      = encrypted_value[3:15]
            ciphertext = encrypted_value[15:-16]
            tag        = encrypted_value[-16:]
            cipher     = AES.new(key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")

        # Legacy: DPAPI-encrypted directly (Chrome < 80)
        import win32crypt
        result = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)
        return result[1].decode("utf-8")

    except Exception as e:
        # Return empty string rather than crashing — some cookies may be
        # encrypted by a different Windows user account and are unreadable.
        print(f"  [decrypt] Could not decrypt a cookie value: {e}")
        return ""


# ── Main export function ───────────────────────────────────────────────────────

def export_cookies() -> dict:
    """
    Read cookies from PROFILE_NAME in CHROME_USER_DATA.

    Returns a Playwright-compatible storage_state dict:
        {"cookies": [...], "origins": []}

    IMPORTANT: Chrome must NOT have the same profile open while you run this,
    because SQLite will be locked. You can:
      - Close that Chrome profile, OR
      - The function automatically copies the DB to a temp file before reading.
    """
    print("[Auth] Reading cookies directly from Chrome (DPAPI + SQLite)...")

    key         = _get_encryption_key(CHROME_USER_DATA)
    cookies_db  = os.path.join(CHROME_USER_DATA, PROFILE_NAME, "Network", "Cookies")

    if not os.path.exists(cookies_db):
        raise FileNotFoundError(
            f"Cookie DB not found at: {cookies_db}\n"
            "Check CHROME_USER_DATA and PROFILE_NAME in chrome_cookie_reader.py"
        )

    # Copy DB to a temp file so we can read it even if Chrome is running
    tmp_db = tempfile.mktemp(suffix=".db")
    shutil.copy2(cookies_db, tmp_db)

    cookies = []
    seen    = set()

    try:
        conn   = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        # Build a WHERE clause for each target domain
        placeholders = " OR ".join(
            ["host_key = ? OR host_key LIKE ?"] * len(COOKIE_DOMAINS)
        )
        params = []
        for domain in COOKIE_DOMAINS:
            # exact match (e.g. ".google.com") + subdomain match (e.g. "mail.google.com")
            params.append(domain)
            params.append(f"%{domain.lstrip('.')}")

        query = f"""
            SELECT name, encrypted_value, host_key, path,
                   expires_utc, is_secure, is_httponly, samesite
            FROM cookies
            WHERE {placeholders}
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

    finally:
        os.unlink(tmp_db)

    for name, enc_val, host_key, path, expires_utc, is_secure, is_httponly, samesite in rows:
        value = _decrypt_cookie_value(enc_val, key)
        if not value:
            continue

        key_tuple = (name, host_key, path)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)

        # Chrome stores time as microseconds since Jan 1 1601.
        # Convert to Unix epoch (seconds since Jan 1 1970).
        expires_unix = None
        if expires_utc and expires_utc > 0:
            expires_unix = int((expires_utc / 1_000_000) - 11_644_473_600)

        samesite_map = {-1: "Unspecified", 0: "Unspecified", 1: "Lax", 2: "Strict", 3: "None"}
        samesite_str = samesite_map.get(samesite, "Lax")

        cookie = {
            "name":     name,
            "value":    value,
            "domain":   host_key,
            "path":     path or "/",
            "secure":   bool(is_secure),
            "httpOnly": bool(is_httponly),
            "sameSite": samesite_str,
        }
        if expires_unix:
            cookie["expires"] = expires_unix

        cookies.append(cookie)

    print(f"[Auth] Exported {len(cookies)} cookies from {PROFILE_NAME}.")

    if len(cookies) == 0:
        print("[Auth] ERROR: No cookies found.")
        print("  → Make sure you are logged into Google in Chrome Profile 23.")
        print("  → Check that CHROME_USER_DATA and PROFILE_NAME are correct.")
        exit(1)

    return {"cookies": cookies, "origins": []}


# ── Quick standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    state   = export_cookies()
    cookies = state["cookies"]
    print(f"\nFirst 5 cookies:")
    for c in cookies[:5]:
        print(f"  {c['domain']:<30} {c['name']:<30} {c['value'][:20]}...")
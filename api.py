from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
import re
import json
import sqlite3
import logging
import urllib.parse
from pathlib import Path
from difflib import SequenceMatcher
from collections import OrderedDict
import time

import httpx
from aiohttp import web as aiohttp_web

# ══════════════════════════════════════════════════════════════════
# الإعدادات
# ══════════════════════════════════════════════════════════════════
PORT = int(os.environ.get("PORT", "8080"))
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "dbs"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════════
# Cache بسيط مع TTL
# ══════════════════════════════════════════════════════════════════

class TTLCache:
    def __init__(self, maxsize: int = 512, ttl: float = 300.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key):
        item = self._cache.get(key)
        if item is None:
            return None
        value, ts = item
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


_quran_cache: TTLCache = TTLCache(maxsize=64, ttl=600.0)
_hadith_cache: TTLCache = TTLCache(maxsize=256, ttl=600.0)
_sunni_cache: TTLCache = TTLCache(maxsize=256, ttl=600.0)


# ══════════════════════════════════════════════════════════════════
# معالجة النص العربي
# ══════════════════════════════════════════════════════════════════

_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
_ALEF_RE = re.compile(r"[أإآٱ]")
_SPACES_RE = re.compile(r"\s+")
_XML_TAGS_RE = re.compile(r"<[^>]+>")


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_RE.sub("ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و")
    text = text.replace("ئ", "ي").replace("ک", "ك").replace("ی", "ي").replace("گ", "ك")
    return text.strip()


# ══════════════════════════════════════════════════════════════════
# البحث القرآني
# ══════════════════════════════════════════════════════════════════

async def search_quran_verses(query: str) -> list[str]:
    cache_key = f"q:{normalize_arabic(query)}"
    cached = _quran_cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"https://api-quran.com/quransql/index.php?text={urllib.parse.quote(query)}&type=search"
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url)
            data = resp.json()
            result = data.get("result") or []
    except Exception as e:
        logger.error(f"Quran API error: {e}")
        result = []

    _quran_cache.set(cache_key, result)
    return result


# ══════════════════════════════════════════════════════════════════
# قاعدة بيانات الأحاديث الشيعية
# ══════════════════════════════════════════════════════════════════

BOOK_DB_MAP: dict[str, str] = {}
VALID_DB_PATHS: list[str] = []
_DB_CONNECTIONS: dict[str, sqlite3.Connection] = {}

BOOK_ALIASES: dict[str, list[str]] = {
    "الكافي": ["الكافي", "كافي", "الكافى"],
    "من لا يحضره الفقيه": [
        "الفقيه", "فقيه", "من لا يحضره الفقيه", "من لا يحضره", "ابن بابويه",
    ],
    "تهذيب الأحكام": [
        "التهذيب", "تهذيب", "تهذيب الاحكام", "تهذيب الأحكام",
    ],
    "الاستبصار": ["الاستبصار", "استبصار", "الإستبصار"],
}

ARABIC_STOP_WORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هو", "هي", "هم", "هن",
    "وهو", "وهي", "وهم", "قد", "قال", "قالت", "ثم", "أن", "إن",
    "لم", "لا", "ما", "بل", "أو", "أم", "كان", "كانت", "يكون",
    "الذي", "التي", "الذين", "اللتي", "ذلك", "تلك", "هذا", "هذه",
    "ليس", "لكن", "حتى", "بعد", "قبل", "عند", "لدى", "كل", "بين",
    "له", "لها", "لهم", "به", "بها", "فيه", "فيها", "منه", "منها",
    "عليه", "عليها", "عنه", "عنها", "إليه", "إليها", "كم", "أي",
}
ARABIC_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")


def smart_keywords(norm_q: str) -> list[str]:
    words = norm_q.split()
    result, seen = [], set()
    for w in words:
        if len(w) < 2 or w in ARABIC_STOP_WORDS:
            continue
        if w not in seen and len(w) >= 3:
            result.append(w)
            seen.add(w)
        stripped = w
        for p in ARABIC_PREFIXES:
            if stripped.startswith(p) and len(stripped) - len(p) >= 3:
                stripped = stripped[len(p):]
                break
        if stripped != w and stripped not in seen and len(stripped) >= 3:
            result.append(stripped)
            seen.add(stripped)
    return result


def get_db_connection(db_path: str) -> sqlite3.Connection:
    if db_path not in _DB_CONNECTIONS:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.create_function("na", 1, lambda s: normalize_arabic(s) if s else "")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        _DB_CONNECTIONS[db_path] = conn
    return _DB_CONNECTIONS[db_path]


def _detect_book_from_db(conn: sqlite3.Connection) -> str | None:
    try:
        cur = conn.cursor()
        cur.execute("SELECT source FROM source_text LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        book = row[0].split(" - ")[0].strip() if " - " in row[0] else row[0].strip()
        book_norm = normalize_arabic(book)
        for key in ("الكافي", "من لا يحضره الفقيه", "تهذيب الأحكام", "الاستبصار"):
            key_norm = normalize_arabic(key)
            if key_norm in book_norm or book_norm.startswith(key_norm[:6]):
                return key
        return book
    except Exception:
        return None


def _build_hadith_db_map():
    global BOOK_DB_MAP, VALID_DB_PATHS
    if not DB_DIR.exists():
        logger.warning(f"مجلد قواعد البيانات غير موجود: {DB_DIR}")
        return
    for fname in os.listdir(DB_DIR):
        if not fname.endswith(".db"):
            continue
        path = str(DB_DIR / fname)
        try:
            conn = get_db_connection(path)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='source_text'"
            )
            if not cur.fetchone():
                continue
            book_key = _detect_book_from_db(conn)
            if path not in VALID_DB_PATHS:
                VALID_DB_PATHS.append(path)
            if book_key and book_key not in BOOK_DB_MAP:
                BOOK_DB_MAP[book_key] = path
        except Exception as e:
            logger.error(f"خطأ في قاعدة بيانات {fname}: {e}")
    logger.info(f"تم تحميل {len(VALID_DB_PATHS)} قاعدة بيانات: {list(BOOK_DB_MAP.keys())}")


def ensure_hadith_dbs():
    if not VALID_DB_PATHS:
        _build_hadith_db_map()


def extract_text_from_xml(xml_str: str) -> str:
    if not xml_str:
        return ""
    text = _XML_TAGS_RE.sub(" ", xml_str)
    return _SPACES_RE.sub(" ", text).strip()


def get_book_name_from_source(source: str) -> str:
    return source.split(" - ")[0].strip() if " - " in source else source.strip()


def detect_hadith_book_filter(text: str) -> tuple[str | None, str]:
    normalized = normalize_arabic(text.strip())
    for book_key, aliases in BOOK_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            alias_norm = normalize_arabic(alias)
            if normalized.startswith(alias_norm):
                rest = text.strip()[len(alias):].strip()
                rest_norm = normalized[len(alias_norm):].strip()
                return book_key, rest if rest else rest_norm
    return None, text.strip()


def _search_hadiths_sync(query: str, book_filter: str | None = None, limit_per_db: int = 6) -> list[dict]:
    ensure_hadith_dbs()
    norm_q = normalize_arabic(query.strip())
    raw_kws = [w for w in norm_q.split() if len(w) >= 2]
    s_kws = smart_keywords(norm_q)
    if not raw_kws:
        return []

    db_paths = (
        [BOOK_DB_MAP[book_filter]]
        if book_filter and book_filter in BOOK_DB_MAP
        else VALID_DB_PATHS
    )
    seen: set = set()
    scored: list[tuple[int, dict]] = []
    total_limit = limit_per_db * len(db_paths)

    for db_path in db_paths:
        try:
            conn = get_db_connection(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, source, info FROM source_text WHERE na(info) LIKE ? LIMIT ?",
                (f"%{norm_q}%", limit_per_db),
            )
            for row in cursor.fetchall():
                key = (db_path, row[0])
                if key not in seen:
                    seen.add(key)
                    scored.append((5, {
                        "book": get_book_name_from_source(row[1]),
                        "source": row[1],
                        "text": extract_text_from_xml(row[2]),
                    }))

            if s_kws and len(scored) < total_limit:
                cond = " AND ".join(["na(info) LIKE ?" for _ in s_kws])
                cursor.execute(
                    f"SELECT id, source, info FROM source_text WHERE {cond} LIMIT ?",
                    [f"%{kw}%" for kw in s_kws] + [limit_per_db],
                )
                for row in cursor.fetchall():
                    key = (db_path, row[0])
                    if key not in seen:
                        seen.add(key)
                        scored.append((4, {
                            "book": get_book_name_from_source(row[1]),
                            "source": row[1],
                            "text": extract_text_from_xml(row[2]),
                        }))

        except Exception as e:
            logger.error(f"خطأ في البحث {db_path}: {e}")

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored]


async def search_hadiths(query: str, book_filter: str | None = None, limit_per_db: int = 6) -> list[dict]:
    cache_key = f"h:{book_filter}:{normalize_arabic(query)}"
    cached = _hadith_cache.get(cache_key)
    if cached is not None:
        return cached

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _THREAD_POOL,
        functools.partial(_search_hadiths_sync, query, book_filter, limit_per_db)
    )

    _hadith_cache.set(cache_key, result)
    return result


# ══════════════════════════════════════════════════════════════════
# الباحث الحديثي السني (درر السنة)
# ══════════════════════════════════════════════════════════════════

DORAR_URL = "https://dorar.net/dorar_api.json"
SUNNI_MAX = 5


def _sunni_parse(items: list) -> list[dict]:
    results = []
    for item in items:
        hadith_raw = item.get("hadith", "")
        hadith_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", hadith_raw)).strip()
        results.append({
            "hadith": hadith_text,
            "rawi": item.get("rawi", ""),
            "mohdith": item.get("mohdith", ""),
            "book": item.get("book", ""),
            "page": item.get("page", ""),
            "grade": item.get("grade", ""),
        })
    return results


async def search_dorar(query: str, page: int = 1) -> dict:
    cache_key = f"sn:{normalize_arabic(query)}:{page}"
    cached = _sunni_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {"q": query, "page": str(page), "spc": "1"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(DORAR_URL, params=params)
            data = resp.json()
    except Exception as e:
        logger.error(f"Dorar API error: {e}")
        return {"error": "تعذّر الاتصال بدرر السنة"}

    ahadith = data.get("ahadith") or {}
    result_items = ahadith.get("result") or []

    if not result_items:
        result = {"error": "لم يُعثر على أحاديث", "results": []}
    else:
        result = {"results": _sunni_parse(result_items[:SUNNI_MAX])}

    _sunni_cache.set(cache_key, result)
    return result


# ══════════════════════════════════════════════════════════════════
# مسارات خادم الويب
# ══════════════════════════════════════════════════════════════════

async def handle_root(request: aiohttp_web.Request) -> aiohttp_web.Response:
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return aiohttp_web.Response(
            text=html_file.read_text(encoding="utf-8"),
            content_type='text/html',
            charset='utf-8'
        )
    return aiohttp_web.Response(text="<h1>index.html غير موجود</h1>", content_type='text/html')


async def handle_quran_search(request: aiohttp_web.Request) -> aiohttp_web.Response:
    q = request.rel_url.query.get('q', '').strip()
    if not q:
        return aiohttp_web.Response(
            text=json.dumps({"results": [], "error": "missing query"}, ensure_ascii=False),
            content_type='application/json', charset='utf-8'
        )
    results = await search_quran_verses(q)
    resp = aiohttp_web.Response(
        text=json.dumps({"results": results[:15]}, ensure_ascii=False),
        content_type='application/json', charset='utf-8'
    )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


async def handle_shia_hadith(request: aiohttp_web.Request) -> aiohttp_web.Response:
    q = request.rel_url.query.get('q', '').strip()
    book = request.rel_url.query.get('book', '').strip() or None
    if not q:
        return aiohttp_web.Response(
            text=json.dumps({"results": [], "error": "missing query"}, ensure_ascii=False),
            content_type='application/json', charset='utf-8'
        )
    results = await search_hadiths(q, book_filter=book)
    resp = aiohttp_web.Response(
        text=json.dumps({"results": results[:10]}, ensure_ascii=False),
        content_type='application/json', charset='utf-8'
    )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


async def handle_sunni_hadith(request: aiohttp_web.Request) -> aiohttp_web.Response:
    q = request.rel_url.query.get('q', '').strip()
    page = int(request.rel_url.query.get('page', '1') or '1')
    if not q:
        return aiohttp_web.Response(
            text=json.dumps({"results": [], "error": "missing query"}, ensure_ascii=False),
            content_type='application/json', charset='utf-8'
        )
    result = await search_dorar(q, page)
    resp = aiohttp_web.Response(
        text=json.dumps(result, ensure_ascii=False),
        content_type='application/json', charset='utf-8'
    )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


async def handle_options(request: aiohttp_web.Request) -> aiohttp_web.Response:
    resp = aiohttp_web.Response(status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


# ══════════════════════════════════════════════════════════════════
# نقطة الانطلاق
# ══════════════════════════════════════════════════════════════════

async def run():
    ensure_hadith_dbs()

    app = aiohttp_web.Application()
    app.router.add_get('/', handle_root)
    app.router.add_get('/api/quran/search', handle_quran_search)
    app.router.add_get('/api/hadith/shia', handle_shia_hadith)
    app.router.add_get('/api/hadith/sunni', handle_sunni_hadith)
    app.router.add_route('OPTIONS', '/{path:.*}', handle_options)

    runner = aiohttp_web.AppRunner(app)
    await runner.setup()
    site = aiohttp_web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"السيرفر يعمل على المنفذ {PORT} - افتح http://localhost:{PORT}")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("تم إيقاف السيرفر.")

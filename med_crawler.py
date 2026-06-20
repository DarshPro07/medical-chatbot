# med_crawler.py
# Requirements: pip install crawl4ai aiohttp beautifulsoup4
import asyncio
import os
import re
import hashlib
from collections import defaultdict
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import aiohttp

# -------------------- Filters (skip junk PDFs) --------------------
EXCLUDE_URL_SUBSTRINGS = [
    "certificate", "payment", "pay.", "/pay", "invoice", "receipt",
    "subscribe", "newsletter", "cookie", "privacy", "terms", "policy",
    "login", "signin", "logout", "account", "register",
    "careers", "jobs", "vacancies", "press", "media",
    "tender", "procurement", "rfp", "rfi", "sponsorship", "donate",
    "adverts", "advert", "marketing", "tracking", "analytics",
]
EXCLUDE_DOMAINS = {
    "payment.mdpi.com",
    "encyclopedia.pub",
}

PDF_REGEXES = [
    r"\.pdf($|\?)",
    r"format=pdf",
    r"type=pdf",
    r"/pdf/",
    r"\bdownload=",
    r"\battachment=",
    r"\bpublicationFile\b",
]

# -------------------- Seeds: add your PDF seeds here --------------------
SEED_URLS = [
    # — Authoritative medical encyclopedias / disease A–Z —
    "https://medlineplus.gov/encyclopedia.html",             # MedlinePlus Medical Encyclopedia (NIH)
    "https://www.msdmanuals.com/professional",               # Merck Manual (Professional)
    "https://www.msdmanuals.com/home",                       # Merck Manual (Consumer)
    "https://www.nhs.uk/conditions/",                        # NHS A–Z Conditions
    "https://www.cdc.gov/diseasesconditions/index.html",     # CDC Diseases & Conditions A–Z
    "https://www.who.int/health-topics",                     # WHO Health Topics A–Z
    "https://www.cancer.gov/publications/pdq",               # NCI PDQ (Patients & Health Professionals)
    "https://rarediseases.info.nih.gov/diseases",            # NIH GARD (Rare diseases)
    "https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN",  # Orphanet (Rare diseases)
    "https://dermnetnz.org/topics",                          # DermNet (Dermatology A–Z)
    "https://wwwnc.cdc.gov/travel/yellowbook/2024",          # CDC Yellow Book (travel medicine)

    # — NCBI Bookshelf: Open-access medical “books” & collections —
    "https://www.ncbi.nlm.nih.gov/books/",                   # NCBI Bookshelf main
    "https://www.ncbi.nlm.nih.gov/books/NBK430685/",         # StatPearls (large disease/procedure compendium)
    "https://www.ncbi.nlm.nih.gov/books/NBK1116/",           # GeneReviews (genetic disorders encyclopedia)
    "https://www.ncbi.nlm.nih.gov/books/NBK201/",            # Clinical Methods (history, physical exam)
    "https://www.ncbi.nlm.nih.gov/books/NBK7627/",           # Medical Microbiology (Baron)
    "https://www.ncbi.nlm.nih.gov/books/NBK279012/",         # Endotext (endocrinology)
    "https://www.ncbi.nlm.nih.gov/books/?term=LiverTox",     # LiverTox (drug-induced liver injury)
    "https://www.ncbi.nlm.nih.gov/books/?term=infectious+disease",
    "https://www.ncbi.nlm.nih.gov/books/?term=clinical+pharmacology",
    "https://www.ncbi.nlm.nih.gov/books/?term=pathophysiology",

    # — Major health systems’ condition libraries (reference-style) —
    "https://www.mayoclinic.org/diseases-conditions",        # Mayo Clinic conditions A–Z
    "https://my.clevelandclinic.org/health/diseases",        # Cleveland Clinic health library
    "https://www.hopkinsmedicine.org/health/conditions-and-diseases",  # Johns Hopkins
    "https://www.stanfordhealthcare.org/medical-conditions.html",      # Stanford Health Care
    "https://www.mountsinai.org/health-library/diseases-conditions",   # Mount Sinai

    # — NICE summaries —
    "https://cks.nice.org.uk/topics/",                       # NICE Clinical Knowledge Summaries (topics)

    # — Open textbooks (anatomy, physiology, microbiology, public health) —
    "https://openstax.org/details/books/anatomy-and-physiology-2e",    # OpenStax A&P 2e
    "https://openstax.org/details/books/microbiology",                  # OpenStax Microbiology
    "https://openstax.org/details/books/biology-2e",                    # OpenStax Biology 2e (background)

    # Open Textbook Library (subject hubs)
    "https://open.umn.edu/opentextbooks/subjects/medicine",
    "https://open.umn.edu/opentextbooks/subjects/nursing",
    "https://open.umn.edu/opentextbooks/subjects/pharmacology",
    "https://open.umn.edu/opentextbooks/subjects/public-health",

    # LibreTexts (Health & Medicine collections)
    "https://med.libretexts.org/Bookshelves",
    "https://bio.libretexts.org/Bookshelves/Microbiology",

    # Pressbooks directory (OER health & medicine)
    "https://pressbooks.directory/subjects/health-medicine/",

    # — Drug references (book-like, authoritative) —
    "https://dailymed.nlm.nih.gov/dailymed/",                # FDA labels (authoritative drug info)
    "https://www.msdmanuals.com/professional/drugs",         # Merck Manual drug reference section

    # — Oncology & specialty encyclopedias (additional) —
    "https://www.cancer.gov/types",                          # NCI cancer types A–Z
    "https://www.hematology.org/education/patients",         # ASH patient ed (hematology)
    "https://www.heart.org/en/health-topics",                # AHA topics (cardio)
    "https://www.thoracic.org/patients/",                    # ATS patient ed (pulmonary)
]
# -------------------- Settings --------------------
DATA_DIR = "data"
MAX_DEPTH = 2
MAX_PAGES_PER_DOMAIN = 50
MAX_TOTAL_PAGES = 1500
MAX_CONCURRENCY = 8
CONNECTOR_LIMIT = 16
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=45)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MedicalPDFCrawler/1.4; +https://example.com/bot)"}
MAX_DOWNLOAD_SIZE_MB = 80
STRICT_PDF = True  # True = prefer real PDFs (check leading %PDF-), but we fallback to Content-Type

# -------------------- Helpers --------------------
def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    return parsed._replace(fragment="").geturl()

def is_http(u: str) -> bool:
    return u.startswith("http://") or u.startswith("https://")

def is_pdfish(u: str) -> bool:
    if not is_http(u):
        return False
    ul = u.lower()
    for bad in EXCLUDE_URL_SUBSTRINGS:
        if bad in ul:
            return False
    netloc = urlparse(u).netloc.lower()
    if netloc in EXCLUDE_DOMAINS:
        return False
    if ul.endswith(".pdf"):
        return True
    return any(re.search(rx, ul) for rx in PDF_REGEXES)

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")

def safe_filename(url: str) -> str:
    p = urlparse(url)
    base = os.path.basename(p.path) or "document.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    base = sanitize(base)
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    root, ext = os.path.splitext(base)
    return f"{sanitize(p.netloc)}_{root}_{h}{ext}"

def extract_links_from_html(base_url: str, html: str) -> set:
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        urls.add(urljoin(base_url, a["href"]))
    for tag in soup.find_all(["source", "iframe", "embed"], src=True):
        urls.add(urljoin(base_url, tag["src"]))
    return {normalize_url(u) for u in urls if is_http(u)}

def extract_links_from_markdown(base_url: str, md: str) -> set:
    urls = set()
    urls.update(re.findall(r"(https?://[^\s)>\]]+)", md))
    urls.update(re.findall(r"<(https?://[^>]+)>", md))
    for rel in re.findall(r'href=[\'"]?(/[^\'" >]+)', md):
        urls.add(urljoin(base_url, rel))
    return {normalize_url(u) for u in urls if is_http(u)}

async def crawl_page(crawler: AsyncWebCrawler, url: str) -> set:
    try:
        res = await crawler.arun(url=url)
        links = set()
        raw_html = getattr(res, "html", None) or getattr(res, "raw_html", None) or ""
        if raw_html:
            links |= extract_links_from_html(url, raw_html)
        md = getattr(res, "markdown", "") or ""
        if md:
            links |= extract_links_from_markdown(url, md)
        return links
    except Exception as exc:
        print(f"[ERR] crawl_page failed for {url}: {exc}")
        return set()

async def head_size(session: aiohttp.ClientSession, url: str) -> int:
    """Try HEAD first; if unavailable, try a small ranged GET to estimate size or get initial bytes."""
    try:
        async with session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status < 400:
                cl = int(resp.headers.get("Content-Length", "0") or "0")
                if cl:
                    return cl
    except Exception:
        pass

    # Fallback: attempt a GET with Range header to get small chunk and/or headers
    try:
        headers = {"Range": "bytes=0-1023"}
        async with session.get(url, headers=headers, allow_redirects=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status < 400:
                cl = int(resp.headers.get("Content-Length", "0") or "0")
                if cl:
                    return cl
                # if server returned partial content, try len of content
                chunk = await resp.content.read(1024)
                if chunk:
                    # if Content-Range header exists, parse it; else use chunk len
                    return int(resp.headers.get("Content-Length", len(chunk)))
    except Exception:
        pass
    return 0

async def download_pdf(session: aiohttp.ClientSession, url: str) -> str | None:
    """Download PDF to DATA_DIR and return path or None."""
    size = await head_size(session, url)
    if size and size > MAX_DOWNLOAD_SIZE_MB * 1024 * 1024:
        print(f"[SKIP - TOO LARGE HEAD] {url} size={size}")
        return None

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, safe_filename(url))
    if os.path.exists(out_path):
        return out_path

    try:
        async with session.get(url, allow_redirects=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status >= 400:
                print(f"[SKIP - HTTP {resp.status}] {url}")
                return None

            cl = int(resp.headers.get("Content-Length", "0") or "0")
            if cl and cl > MAX_DOWNLOAD_SIZE_MB * 1024 * 1024:
                print(f"[SKIP - TOO LARGE CONTENT-LENGTH] {url} size={cl}")
                return None

            max_bytes = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024
            written = 0
            # Read first chunk to validate PDF signature (if STRICT_PDF)
            first_chunk = await resp.content.read(8192)
            if not first_chunk:
                print(f"[SKIP - EMPTY] {url}")
                return None

            content_type = resp.headers.get("Content-Type", "").lower()
            looks_like_pdf = first_chunk.startswith(b"%PDF-") or ("pdf" in content_type)

            if STRICT_PDF and not looks_like_pdf:
                # try to be lenient: if headers indicate pdf, accept; otherwise skip
                print(f"[SKIP - NOT PDF] {url} content-type={content_type}")
                return None

            # Write file (we already read the first chunk)
            with open(out_path, "wb") as f:
                f.write(first_chunk)
                written += len(first_chunk)
                if written > max_bytes:
                    print(f"[SKIP - EXCEEDED MAX BYTES AFTER FIRST CHUNK] {url}")
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    return None

                async for chunk in resp.content.iter_chunked(1 << 14):
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        print(f"[SKIP - EXCEEDED MAX BYTES WHILE WRITING] {url}")
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass
                        return None
                    f.write(chunk)

        return out_path
    except Exception as exc:
        print(f"[ERR] download failed {url}: {exc}")
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return None

# -------------------- Crawl + download --------------------
async def crawl_and_download_to_data(seeds: list):
    visited: set = set()
    scheduled: set = set()
    domain_count = defaultdict(int)

    q = asyncio.Queue()
    for s in seeds:
        q.put_nowait((normalize_url(s), 0))

    conn = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with AsyncWebCrawler() as crawler, aiohttp.ClientSession(headers=HEADERS, connector=conn) as session:
        async def worker():
            while True:
                try:
                    url, depth = await q.get()
                except asyncio.CancelledError:
                    break

                if url in visited or not is_http(url):
                    q.task_done()
                    continue

                dom = urlparse(url).netloc.lower()
                if domain_count[dom] >= MAX_PAGES_PER_DOMAIN or len(visited) >= MAX_TOTAL_PAGES:
                    q.task_done()
                    continue

                visited.add(url)
                domain_count[dom] += 1

                async with sem:
                    links = await crawl_page(crawler, url)

                # Schedule downloads:
                pdfish_links = [u for u in links if is_pdfish(u)]

                # --- IMPORTANT: if the current URL itself is a PDF, include it ---
                todo_candidates = pdfish_links.copy()
                if is_pdfish(url):
                    todo_candidates.insert(0, url)

                todo = [u for u in todo_candidates if u not in scheduled]
                for u in todo:
                    scheduled.add(u)

                if todo:
                    print(f"[DOWNLOAD SCHEDULE] {len(todo)} files from {url}")
                    tasks = [download_pdf(session, u) for u in todo]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for u, path in zip(todo, results):
                        if isinstance(path, str) and path:
                            print(f"[OK] {path}  ← from {u}")
                        else:
                            print(f"[SKIP] {u}")

                # Enqueue deeper crawl (non-pdf links)
                if depth < MAX_DEPTH:
                    crawlable = [u for u in links if not is_pdfish(u)]
                    for u in crawlable:
                        if u not in visited:
                            q.put_nowait((u, depth + 1))

                q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)]
        await q.join()
        for w in workers:
            w.cancel()

# -------------------- Entrypoint --------------------
async def main1():
    print("[INIT] Crawling and downloading hospital-related PDFs to ./data ...")
    await crawl_and_download_to_data(SEED_URLS)
    print("[DONE]")
    from create_memory_for_llm import main
    main()

if __name__ == "__main__":
    asyncio.run(main1())

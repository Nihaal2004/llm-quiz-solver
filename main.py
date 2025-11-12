# main.py
import mimetypes, base64
import os, time, hmac, asyncio, re, io, json, uuid, logging, base64
from typing import Any, Optional, Tuple, List
from urllib.parse import urljoin

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright
import httpx
import pdfplumber
import pandas as pd

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("solver")

app = FastAPI(title="LLM Quiz Solver")
SECRET = os.environ.get("QUIZ_SECRET", "")

def ct_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/solve")
async def solve(request: Request, background: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")

    if not isinstance(email, str) or not isinstance(secret, str) or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing fields")

    if not SECRET or not ct_equal(secret, SECRET):
        raise HTTPException(status_code=403, detail="Invalid secret")

    tid = uuid.uuid4().hex[:8]
    t0 = time.monotonic()
    if os.getenv("DISABLE_WORKER") == "1":   # <-- add this
        return JSONResponse({"status": "accepted"})
    background.add_task(run_chain, tid, email, secret, url, t0)
    return JSONResponse({"status": "accepted"})

# ---------- core loop ----------
async def run_chain(tid: str, email: str, secret: str, first_url: str, t0: float):
    deadline = t0 + 180.0
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=False)
        page = await context.new_page()
        current = first_url
        try:
            while current and time.monotonic() < deadline:
                log.info(json.dumps({"ev":"visit","tid":tid,"url":current}))
                await page.goto(current, wait_until="networkidle")
                await page.wait_for_load_state("domcontentloaded")
                # give late scripts a moment
                try:
                    await page.wait_for_function("document.body && document.body.innerText.length > 0", timeout=2000)
                except Exception:
                    pass

                text, html, decoded = await extract_instructions(page)
                submit_url = discover_submit_url(page, text, html, decoded)
                answer = await compute_answer(page, text, html, decoded, deadline)

                ok, maybe_next = await submit_answer(submit_url, email, secret, current, answer)
                log.info(json.dumps({"ev":"submit","tid":tid,"ok":ok,"next":maybe_next,"ans_type":type(answer).__name__}))

                if not ok and not maybe_next and time.monotonic() < deadline - 10:
                    answer = await compute_answer(page, text, html, decoded, deadline, retry=True)
                    ok, maybe_next = await submit_answer(submit_url, email, secret, current, answer)
                    log.info(json.dumps({"ev":"retry","tid":tid,"ok":ok,"next":maybe_next,"ans_type":type(answer).__name__}))

                current = maybe_next

        except Exception as e:
            log.info(json.dumps({"ev":"error","tid":tid,"msg":str(e)}))
        finally:
            await context.close()
            await browser.close()
            log.info(json.dumps({"ev":"done","tid":tid,"dur_s":round(time.monotonic()-t0,3)}))

# ---------- DOM + parsing helpers ----------
async def extract_instructions(page) -> Tuple[str, str, str]:
    """Return (visible_text, html, decoded_from_atob_or_pre)."""
    text = await page.evaluate("document.body ? document.body.innerText : ''")
    html = await page.content()

    # gather <pre> blocks that often hold JSON hints
    pre_text = await page.evaluate("""
        () => Array.from(document.querySelectorAll('pre'))
              .map(n => n.innerText).join("\\n\\n")
    """)

    # decode any atob(`...`) literal in scripts
    scripts = await page.evaluate("""
        () => Array.from(document.scripts).map(s => s.textContent || "")
    """)
    decoded_chunks: List[str] = []
    for src in scripts:
        for m in re.finditer(r"atob\\(`([A-Za-z0-9+/=\\n\\r]+)`\\)", src):
            b64 = m.group(1).replace("\\n","").replace("\\r","")
            try:
                decoded_chunks.append(base64.b64decode(b64).decode("utf-8", "ignore"))
            except Exception:
                pass

    decoded = "\n\n".join(([pre_text] if pre_text else []) + decoded_chunks)
    return text, html, decoded

def discover_submit_url(page, text: str, html: str, decoded: str) -> str:
    # priority 1: explicit "... submit to https://... " phrasing
    for blob in (decoded, text, html):
        m = re.search(r"https?://[^\s\"'>]+/submit[^\s\"'>]*", blob or "", re.I)
        if m:
            return m.group(0)
    # fallback: any URL immediately after the keyword "submit"
    for blob in (decoded, text):
        m = re.search(r"submit[^:\n]*[:]\s*(https?://[^\s\"'>]+)", blob or "", re.I)
        if m:
            return m.group(1)
    # last resort: DOM anchors containing "submit"
    # (we resolve relative URLs to absolute)
    # this runs in page context to gather hrefs
    # note: use sync wrapper via run_sync? not needed; keep simple via regex above
    raise RuntimeError("Submit URL not found")

def abs_url(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href

# ---------- solvers ----------
async def compute_answer(page, text: str, html: str, decoded: str, deadline: float, retry: bool=False) -> Any:
    # bail out if almost out of time
    if time_left(deadline) < 5:
        return "timeout"

    # A) PDF on the page → sum "value" column on page 2
    pdf_url = await first_pdf_href(page)
    if pdf_url and time_left(deadline) > 10:
        s = await sum_value_col_from_pdf(pdf_url, deadline)
        if s is not None:
            return s

    # B) CSV/XLSX links
    data_link = await first_data_link(page)
    if data_link:
        kind, url = data_link
        if kind == "csv":
            return await sum_value_from_csv(url, deadline)
        if kind == "xlsx":
            return await sum_value_from_xlsx(url, deadline)

    # C) Visible HTML tables (choose the one with a 'value' column or highest numeric density)
    try:
        has_tables = await page.evaluate("document.querySelectorAll('table').length")
        if has_tables:
            rows_list = await page.evaluate("""
                () => Array.from(document.querySelectorAll('table')).map(t =>
                    Array.from(t.querySelectorAll('tr')).map(tr =>
                        Array.from(tr.children).map(td => td.innerText.trim())
                    )
                )
            """)
            best = pick_table_sum(rows_list)
            if best is not None:
                return float(best)
    except Exception:
        pass

    # D) Text block labeled "value"
    for blob in (decoded, text):
        blk = extract_value_block(blob or "")
        if blk:
            nums = [to_num(x) for x in re.findall(r"[-+]?\d*\.?\d+", blk)]
            total = sum(n for n in nums if n is not None)
            return float(total)

    # E) If instructions ask for a chart, return a PNG data URI
    for blob in (decoded, text):
        if blob and re.search(r"\b(chart|plot|bar\s*chart|visuali[sz]e)\b", blob, re.I):
            nums = [to_num(x) for x in re.findall(r"[-+]?\d*\.?\d+", blob)]
            nums = [n for n in nums if n is not None]
            if len(nums) >= 2:
                return render_bar_chart_png(nums[:12])  # cap bars for safety

    return "unable_to_determine"

def pick_table_sum(all_tables: List[List[List[str]]]) -> Optional[float]:
    best_score, best_sum = -1.0, None
    for tbl in all_tables:
        if not tbl or len(tbl) < 2:
            continue
        header = [c.strip() for c in tbl[0]]
        # find value column (exact or fuzzy)
        def ix():
            for i, h in enumerate(header):
                hlow = h.lower()
                if hlow == "value" or "value" in hlow or hlow.endswith("value"):
                    return i
            return None
        vidx = ix()
        if vidx is None:
            # choose numeric-dense column if no 'value'
            cand_idx, cand_density = None, -1.0
            for j in range(len(header)):
                nums = 0
                for row in tbl[1:]:
                    if j < len(row):
                        if re.search(r"\d", str(row[j])):
                            nums += 1
                density = nums / max(1, len(tbl)-1)
                if density > cand_density:
                    cand_idx, cand_density = j, density
            vidx = cand_idx
        if vidx is None:
            continue
        total = 0.0
        nums = 0
        for row in tbl[1:]:
            cell = row[vidx] if vidx < len(row) else None
            if isinstance(cell, str):
                cell = cell.replace(",", "").strip()
            try:
                total += float(cell)
                nums += 1
            except Exception:
                pass
        score = nums  # simple: more numeric rows is better
        if score > best_score and nums > 0:
            best_score, best_sum = score, total
    return best_sum

async def first_pdf_href(page) -> Optional[str]:
    try:
        links = page.locator('a')
        n = await links.count()
        best = None
        for i in range(n):
            href = await links.nth(i).get_attribute("href")
            text = (await links.nth(i).inner_text()).lower() if await links.nth(i).is_visible() else ""
            if not href:
                continue
            if ".pdf" in href.lower():
                # prefer anchors labeled "download" or "data"
                score = 2 if ("download" in text or "data" in text) else 1
                full = await page.evaluate("(u) => new URL(u, location.href).href", href)
                if not best or score > best[0]:
                    best = (score, full)
        return best[1] if best else None
    except Exception:
        return None

async def first_data_link(page) -> Optional[Tuple[str, str]]:
    try:
        links = page.locator('a')
        n = await links.count()
        for i in range(n):
            href = await links.nth(i).get_attribute("href")
            if not href:
                continue
            hl = href.lower()
            if hl.endswith(".csv") or ".csv?" in hl:
                full = await page.evaluate("(u) => new URL(u, location.href).href", href)
                return ("csv", full)
            if hl.endswith(".xlsx") or hl.endswith(".xls") or ".xlsx?" in hl or ".xls?" in hl:
                full = await page.evaluate("(u) => new URL(u, location.href).href", href)
                return ("xlsx", full)
    except Exception:
        pass
    return None

async def sum_value_col_from_pdf(url: str, deadline: float) -> Optional[float]:
    timeout = min(30, max(5, int(deadline - time.monotonic() - 5)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = io.BytesIO(r.content)
    try:
        with pdfplumber.open(data) as pdf:
            idx = 1  # page 2
            if idx >= len(pdf.pages):
                return None
            tbl = pdf.pages[idx].extract_table()
            if not tbl or len(tbl) < 2:
                return None
            header = [h.strip().lower() if isinstance(h, str) else "" for h in tbl[0]]
            if "value" not in header:
                return None
            vidx = header.index("value")
            total = 0.0
            for row in tbl[1:]:
                cell = row[vidx] if vidx < len(row) else None
                if isinstance(cell, str):
                    cell = cell.replace(",", "").strip()
                try:
                    total += float(cell)
                except Exception:
                    pass
            return total
    except Exception:
        return None

async def sum_value_from_csv(url: str, deadline: float) -> Optional[float]:
    timeout = min(30, max(5, int(deadline - time.monotonic() - 5)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    col = pick_value_column(df.columns)
    if col:
        return float(pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0).sum())
    return None

async def sum_value_from_xlsx(url: str, deadline: float) -> Optional[float]:
    timeout = min(30, max(5, int(deadline - time.monotonic() - 5)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))
    col = pick_value_column(df.columns)
    if col:
        return float(pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0).sum())
    return None

def pick_value_column(cols) -> Optional[str]:
    for c in cols:
        cl = str(c).strip().lower()
        if cl == "value" or "value" in cl:
            return c
    return None

def extract_value_block(text: str) -> Optional[str]:
    m = re.search(r"value[^:\n]*[:\n](.+?)(?:\n\n|\Z)", text, re.I | re.S)
    return m.group(1) if m else None

def to_num(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None
def time_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())

def data_uri_from_bytes(b: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"

def render_bar_chart_png(values: list[float], labels: Optional[list[str]] = None) -> str:
    import matplotlib.pyplot as plt  # lazy import
    fig = plt.figure()
    x = range(len(values))
    plt.bar(x, values)
    if labels and len(labels) == len(values):
        plt.xticks(list(x), labels, rotation=0)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return data_uri_from_bytes(buf.getvalue(), "image/png")


async def submit_answer(submit_url: str, email: str, secret: str, task_url: str, answer: Any) -> Tuple[bool, Optional[str]]:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(submit_url, json={
                "email": email,
                "secret": secret,
                "url": task_url,
                "answer": answer
            })
            resp.raise_for_status()
            data = resp.json()
            ok = bool(data.get("correct", False))
            next_url = data.get("url")
            return ok, next_url if isinstance(next_url, str) else None
        except httpx.HTTPStatusError as e:
            # surface server status in logs but keep flow alive
            log.info(json.dumps({"ev":"submit_http_error","code":e.response.status_code}))
            return False, None

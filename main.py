# main.py
import os, time, hmac, asyncio, re, io, json, uuid, logging
from typing import Any, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright
import httpx
import pdfplumber
import pandas as pd

# ---- logging: single-line JSON for easy reading in Railway ----
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
    log.info(json.dumps({"ev":"accepted","tid":tid,"email":email,"url":url,"t":t0}))
    background.add_task(run_chain, tid, email, secret, url, t0)
    return JSONResponse({"status": "accepted"})

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

                text = await page.evaluate("document.body.innerText")
                html = await page.content()

                submit_url = find_submit_url(text, html)
                answer = await compute_answer(page, text, html, deadline)
                ok, maybe_next = await submit_answer(submit_url, email, secret, current, answer)
                log.info(json.dumps({"ev":"submit","tid":tid,"ok":ok,"next":maybe_next,"ans_type":type(answer).__name__}))

                if not ok and not maybe_next and time.monotonic() < deadline - 10:
                    answer = await compute_answer(page, text, html, deadline, retry=True)
                    ok, maybe_next = await submit_answer(submit_url, email, secret, current, answer)
                    log.info(json.dumps({"ev":"retry","tid":tid,"ok":ok,"next":maybe_next,"ans_type":type(answer).__name__}))

                current = maybe_next

        except Exception as e:
            log.info(json.dumps({"ev":"error","tid":tid,"msg":str(e)}))
        finally:
            await context.close()
            await browser.close()
            log.info(json.dumps({"ev":"done","tid":tid,"dur_s":round(time.monotonic()-t0,3)}))

def find_submit_url(text: str, html: str) -> str:
    m = re.search(r"https?://[^\s\"'>]+/submit[^\s\"'>]*", text, re.I)
    if m: return m.group(0)
    m = re.search(r"https?://[^\s\"'>]+/submit[^\s\"'>]*", html, re.I)
    if m: return m.group(0)
    m = re.search(r"submit[^:\n]*:\s*(https?://[^\s\"'>]+)", text, re.I)
    if m: return m.group(1)
    raise RuntimeError("Submit URL not found")

async def compute_answer(page, text: str, html: str, deadline: float, retry: bool=False) -> Any:
    pdf_url = await first_pdf_href(page)
    if pdf_url:
        s = await sum_value_col_from_pdf(pdf_url)
        if s is not None:
            return s

    try:
        has_table = await page.evaluate("document.querySelector('table') !== null")
        if has_table:
            rows = await page.evaluate("""
                () => {
                  const t = document.querySelector('table');
                  const rs = [...t.querySelectorAll('tr')].map(tr => [...tr.children].map(td => td.innerText.trim()));
                  return rs;
                }
            """)
            if rows and len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                target = next((c for c in df.columns if c.strip().lower() == "value"), None)
                if target:
                    vals = pd.to_numeric(df[target].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
                    return float(vals.sum())
    except Exception:
        pass

    blk = extract_value_block(text)
    if blk:
        nums = [to_num(x) for x in re.findall(r"[-+]?\d*\.?\d+", blk)]
        total = sum(n for n in nums if n is not None)
        return float(total)

    return "unable_to_determine"

async def first_pdf_href(page) -> Optional[str]:
    try:
        links = page.locator('a')
        n = await links.count()
        for i in range(n):
            href = await links.nth(i).get_attribute("href")
            if href and ".pdf" in href.lower():
                full = await page.evaluate("(u) => new URL(u, location.href).href", href)
                return full
    except Exception:
        pass
    return None

async def sum_value_col_from_pdf(url: str) -> Optional[float]:
    async with httpx.AsyncClient(timeout=60) as client:
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

def extract_value_block(text: str) -> Optional[str]:
    m = re.search(r"value[^:\n]*[:\n](.+?)(?:\n\n|\Z)", text, re.I | re.S)
    return m.group(1) if m else None

def to_num(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

async def submit_answer(submit_url: str, email: str, secret: str, task_url: str, answer: Any) -> Tuple[bool, Optional[str]]:
    async with httpx.AsyncClient(timeout=30) as client:
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

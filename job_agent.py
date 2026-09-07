#!/usr/bin/env python3
"""
job_agent.py

Täglicher Job-Sucher:
- Fragt die BA-Jobsuche-API für vordefinierte Queries ab.
- Vergleicht mit seen_jobs.json und erkennt neue Angebote.
- Optional: Fragt ein LLM (Gemini / Generative API) zur Relevanz- und Anschreiben-Generierung ab.
- Sendet Notifications per CallMeBot WhatsApp.
- Speichert neue IDs in seen_jobs.json (atomic write).
"""

import os
import sys
import json
import time
import hashlib
import logging
import tempfile
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Config
SEEN_FILE = "seen_jobs.json"
DRAFTS_DIR = "drafts"
BA_API_URL = "https://jobsuche.api.bund.de/jobs"  # base, params per API docs
QUERIES = [
    {"suchbegriff": "Werkstudent Maschinenbau", "wo": "Nürnberg", "umkreis": 40},
    {"suchbegriff": "Bachelorarbeit Maschinenbau", "wo": "Nürnberg", "umkreis": 40},
]
ANGEBOOTSART = "1;34"     # normale Anstellung + Praktikum/Trainee
VEROEFFENTLICHT_SEIT = 3  # Tage

# Environment / Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATIVE_API_URL = os.getenv("GENERATIVE_API_URL")  # optional override
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
WHATSAPP_PHONE = os.getenv("WHATSAPP_PHONE")  # e.g. 4915123456789 (no +)

# Personal profile (must be edited by user)
MEIN_PROFIL = {
    "name": "Max Mustermann",
    "semester": "6",
    "studiengang": "Maschinenbau",
    "schwerpunkte": "Konstruktion, FEM, CAD (SolidWorks)",
    "stärken": "sorgfältiges Arbeiten, Teamfähigkeit, schnelle Einarbeitung",
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("job_agent")

# HTTP session with retries
def make_session(retries: int = 3, backoff_factor: float = 0.8) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "job-agent/1.0"})
    return session

session = make_session()

def load_seen() -> List[str]:
    if not os.path.exists(SEEN_FILE):
        logger.info("seen_jobs.json nicht gefunden, erstelle neue Datei.")
        return []
    try:
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Fehler beim Laden von seen_jobs.json: %s", e)
        return []

def atomic_write(path: str, data: Any) -> None:
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirn, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
    logger.info("Atomar geschrieben: %s", path)

def job_unique_id(job: Dict[str, Any]) -> str:
    # Versuche mit vorhandener ID, sonst Hash aus Titel+Firma+Ort+Datum
    if "id" in job and job["id"]:
        return str(job["id"])
    parts = [
        job.get("title", ""),
        job.get("unternehmen", job.get("company", "")),
        job.get("ort", job.get("location", "")),
        str(job.get("veroeffentlicht", "")),
    ]
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def fetch_jobs_for(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = {
        "suchbegriff": query.get("suchbegriff"),
        "wo": query.get("wo"),
        "umkreis": query.get("umkreis"),
        "angebotsart": ANGEBOOTSART,
        "veroeffentlichtseit": VEROEFFENTLICHT_SEIT,
        "seitenanzahl": 1,
        "seitengröße": 50,
    }
    try:
        r = session.get(BA_API_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        # API Struktur variiert; passe ggf. an. Erwartet: list unter 'results' oder direkt als liste.
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            # fallback: try common keys
            results = data.get("jobs") or data.get("stellen") or []
        logger.info("Query '%s' lieferte %d Einträge", params["suchbegriff"], len(results))
        return results
    except Exception as e:
        logger.error("Fehler bei BA-API Anfrage: %s", e)
        return []

def call_generative(prompt: str, max_tokens: int = 400) -> Optional[str]:
    """
    Generative LLM call (optional). Wenn GEMINI_API_KEY gesetzt ist und GENERATIVE_API_URL
    vorhanden (oder Standard), wird ein POST ausgeführt. Die genaue API-Antwortstruktur
    kann je nach Endpoint variieren -> flexible Auslese.
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        logger.debug("GEMINI_API_KEY nicht gesetzt - generatives Modell übersprungen.")
        return None
    url = GENERATIVE_API_URL or "https://generative.googleapis.com/v1beta2/models/text-bison-001:generate"
    payload = {"prompt": prompt}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = session.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        # Versuche flexible Extraktion
        if isinstance(j, dict):
            # common shapes: { "candidates": [{"content": "..."}] } or { "output": [{"content":"..."}] } or { "reply": "..." }
            if "candidates" in j and j["candidates"]:
                return j["candidates"][0].get("content") or j["candidates"][0].get("text")
            if "output" in j and j["output"]:
                o = j["output"][0]
                return o.get("content") or o.get("text") or o.get("response")
            # generic text field
            for k in ("text", "reply", "output_text", "response"):
                if k in j:
                    return j[k]
        # fallback: stringify
        return json.dumps(j)
    except Exception as e:
        logger.warning("Fehler beim Aufruf des generativen Modells: %s", e)
        return None

def is_relevant_by_gemini(job: Dict[str, Any]) -> bool:
    # Fallback: keyword matching if no LLM available
    title = (job.get("title") or job.get("stelle") or "").lower()
    description = (job.get("beschreibung") or job.get("description") or "").lower()
    if "maschinenbau" in title or "maschinenbau" in description:
        basic = True
    else:
        basic = False

    if not GEMINI_API_KEY:
        logger.debug("Kein LLM-Key: nutze einfache Keyword-Filter → %s", basic)
        return basic

    prompt = (
        "Du bist ein Assistent, der Job-Anzeigen für einen Maschinenbau-Studierenden klassifiziert. "
        "Entscheide, ob die Anzeige RELEVANT ist für 'Werkstudent Maschinenbau' oder 'Bachelorarbeit Maschinenbau'.\n\n"
        f"TITEL: {job.get('title','')}\n\nBESCHREIBUNG: {job.get('description','')[:800]}\n\n"
        "Gib nur 'JA' oder 'NEIN' zurück und kurz (ein Wort) optional einen Grund."
    )
    resp = call_generative(prompt, max_tokens=80)
    if not resp:
        return basic
    resp_l = resp.strip().lower()
    if resp_l.startswith("ja") or resp_l.startswith("yes") or "ja" in resp_l.split():
        return True
    if resp_l.startswith("nein") or resp_l.startswith("no"):
        return False
    # fallback
    return basic

def generate_cover_letter(job: Dict[str, Any], profile: Dict[str, str]) -> str:
    # If LLM available, generate personalized draft, otherwise simple template
    if GEMINI_API_KEY:
        prompt = (
            "Erstelle ein kurzes Bewerbungsanschreiben (Deutsch) als Werkstudent/Bachelorand in Maschinenbau. "
            "Maximal ca. 220 Wörter. Nutze dieses Profil:\n"
            + json.dumps(profile, ensure_ascii=False)
            + "\n\nStelleninfo:\n"
            f"Titel: {job.get('title','')}\nFirma: {job.get('company') or job.get('unternehmen','')}\nOrt: {job.get('location') or job.get('ort','')}\nAufgabe: {job.get('description','')[:800]}\n\n"
            "Erzeuge einen höflichen, prägnanten Entwurf, der personalisiert wirkt und konkrete Skills erwähnt."
        )
        resp = call_generative(prompt, max_tokens=600)
        if resp:
            return resp.strip()
    # fallback template
    return (
        f"Sehr geehrte Damen und Herren,\n\n"
        f"mit großem Interesse habe ich die Stelle '{job.get('title','')}' bei {job.get('company') or job.get('unternehmen','')} in {job.get('location') or job.get('ort','')} gelesen. "
        f"Ich studiere {profile.get('studiengang')} (Semester {profile.get('semester')}) mit Schwerpunkt {profile.get('schwerpunkte')}. "
        f"Meine Erfahrungen in Konstruktion und CAD (z. B. SolidWorks) sowie ein sicherer Umgang mit FEM-Methoden machen mich zu einer passenden Besetzung für diese Aufgabe.\n\n"
        f"Über die Einladung zu einem persönlichen Gespräch freue ich mich sehr.\n\nMit freundlichen Grüßen\n{profile.get('name')}"
    )

def send_whatsapp_message(text: str) -> bool:
    if not CALLMEBOT_APIKEY or not WHATSAPP_PHONE:
        logger.warning("CallMeBot oder Telefonnummer nicht konfiguriert; WhatsApp übersprungen.")
        return False
    url = "https://api.callmebot.com/whatsapp.php"
    params = {
        "phone": WHATSAPP_PHONE,
        "text": text,
        "apikey": CALLMEBOT_APIKEY,
    }
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        logger.info("WhatsApp gesendet (CallMeBot).")
        return True
    except Exception as e:
        logger.warning("Fehler beim Senden per WhatsApp: %s", e)
        return False

def main():
    logger.info("Starte Job-Agent")
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    seen = set(load_seen())
    new_seen = set(seen)
    all_new_jobs = []

    for q in QUERIES:
        jobs = fetch_jobs_for(q)
        for job in jobs:
            uid = job_unique_id(job)
            if uid in seen:
                continue
            # preliminary accept if contains Maschinenbau OR LLM says yes
            if not is_relevant_by_gemini(job):
                logger.info("Gefiltert als nicht relevant: %s", job.get("title"))
                new_seen.add(uid)  # mark as seen to avoid repeated checks for same non-relevant ad
                continue
            # prepare notification payload
            title = job.get("title") or job.get("stelle") or "n/a"
            company = job.get("company") or job.get("unternehmen") or "n/a"
            location = job.get("location") or job.get("ort") or "n/a"
            link = job.get("link") or job.get("url") or job.get("anzeigeURL") or ""
            msg = f"Neue Stelle: {title}\n{company} — {location}\n{link}"
            # send notification
            send_whatsapp_message(msg)
            # generate cover letter draft for top 2 later (collect first)
            all_new_jobs.append({"uid": uid, "job": job})
            new_seen.add(uid)

    if all_new_jobs:
        # sort/pick top 2 by heuristics (e.g., recent first) — fallback: original order
        top2 = all_new_jobs[:2]
        for idx, entry in enumerate(top2, start=1):
            job = entry["job"]
            draft = generate_cover_letter(job, MEIN_PROFIL)
            fname = os.path.join(DRAFTS_DIR, f"cover_{entry['uid'][:10]}_{idx}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"// Job: {job.get('title','')} — {job.get('company') or job.get('unternehmen','')}\n\n")
                f.write(draft)
            # send draft via WhatsApp (short version)
            short = f"Anschreiben-Entwurf #{idx} für '{job.get('title','')}' bei {job.get('company') or job.get('unternehmen','')}:\n\n{draft[:800]}"
            send_whatsapp_message(short)
            logger.info("Anschreiben erstellt: %s", fname)

    # persist seen list atomically
    try:
        atomic_write(SEEN_FILE, sorted(list(new_seen)))
        logger.info("seen_jobs.json aktualisiert (%d Einträge)", len(new_seen))
    except Exception as e:
        logger.exception("Fehler beim Schreiben von seen_jobs.json: %s", e)

    logger.info("Job-Agent fertig.")

if __name__ == "__main__":
    main()

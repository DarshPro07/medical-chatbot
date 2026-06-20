import os
import re
import json
import html
import requests
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Optional browser geolocation (no sidebar UI)
try:
    from streamlit_js_eval import get_geolocation  # pip install streamlit-js-eval
except Exception:
    get_geolocation = None

# LangChain / RAG
from tavily import TavilyClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# OSM / mapping (free)
import overpy
import folium
from streamlit_folium import st_folium


# ========================= Config / Tunables =========================
DB_FAISS_PATH = "vectorstore/db_faiss"
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))

# Retrieval knobs
TOP_K = int(os.getenv("TOP_K", "6"))
FETCH_K = int(os.getenv("FETCH_K", "40"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.5"))  # 0=diversity, 1=similarity

# LLM knobs
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.5"))

# Web snippet clipping
WEB_SNIPPET_CHARS = int(os.getenv("WEB_SNIPPET_CHARS", "360"))

DISCLAIMER_TEXT = (
    "Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only and "
    "should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider."
)

# Medical-only guard
MED_POSITIVE_HINTS = [
    "symptom", "symptoms", "diagnosis", "treatment", "medicine", "drug", "dose",
    "health", "disease", "disorder", "syndrome", "vaccine", "cancer", "doctor",
    "hospital", "clinic", "emergency", "er ", "a&e", "urgent care",
    "therap", "oncology", "cardio", "neuro", "derma",
    "infection", "pathogen", "lab test", "blood test", "biopsy", "guideline",
    "contraindication", "side effect", "adverse", "pharma", "anatomy", "physiology",
    "microbiology", "immunology", "gene", "genetic", "pregnan", "pediatric"
]
NON_MED_RED_FLAGS = [
    "stock", "share", "nifty", "sensex", "nasdaq", "bitcoin", "crypto", "price of",
    "python", "java", "javascript", "coding", "program", "script", "algorithm",
    "movie", "song", "lyrics", "recipe", "cooking", "football", "cricket",
    "president", "election", "politics", "gdp", "budget", "weather", "forecast"
]
WEB_INTENTS = [
    "best", "top", "latest", "today", "news", "near me", "nearby", "open now",
    "price", "prices", "ranking", "ranked", "review", "reviews",
    "compare", " vs ", "address", "hours", "contact", "phone",
    "buy", "sell", "market", "hospit", "clinic", "update", "2025", "closest", "nearest"
]
_UNSURE_RE = re.compile(
    r"(i (don't|do not) know|not sure|cannot find|no information|insufficient information|i'm not able to find)",
    re.IGNORECASE
)
# =====================================================================


# =========================== Caching layers ===========================
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY. Add it to your .env")
    return TavilyClient(api_key=api_key)

@st.cache_resource(show_spinner=False)
def get_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not set in .env.")
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=GROQ_TEMPERATURE,
        api_key=GROQ_API_KEY or "REPLACE_ME_WITH_ENV",
    )

@st.cache_data(ttl=1800, show_spinner=False)
def tavily_search_cached(query: str, depth: str, include_domains_tuple: tuple, max_results: int):
    client = get_tavily_client()
    include_domains = list(include_domains_tuple) if include_domains_tuple else None
    return client.search(query=query, search_depth=depth, include_domains=include_domains, max_results=max_results)

@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str:
    try:
        headers = {"User-Agent": "MediMind/1.0 (contact: you@example.com)"}
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon},
            headers=headers, timeout=6
        )
        if not r.ok:
            return f"{lat:.4f},{lon:.4f}"
        data = r.json()
        addr = data.get("address", {}) or {}
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county")
        region = addr.get("state")
        country = addr.get("country")
        parts = [p for p in [city, region, country] if p]
        return ", ".join(parts) if parts else f"{lat:.4f},{lon:.4f}"
    except Exception:
        return f"{lat:.4f},{lon:.4f}"

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_place(place: str):
    """Forward geocode city/region/country -> (lat, lon) using Nominatim."""
    try:
        headers = {"User-Agent": "MediMind/1.0 (contact: you@example.com)"}
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"format": "json", "q": place, "limit": 1},
            headers=headers, timeout=6
        )
        if not r.ok:
            return None
        arr = r.json()
        if not arr:
            return None
        lat = float(arr[0]["lat"]); lon = float(arr[0]["lon"])
        return lat, lon
    except Exception:
        return None
# =====================================================================


# ============================ Utilities ==============================
def is_clearly_non_medical(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in NON_MED_RED_FLAGS) and not any(k in ql for k in MED_POSITIVE_HINTS)

def should_force_web_search(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in WEB_INTENTS)

def is_low_confidence_rag(answer_text: str, context_docs) -> bool:
    if not answer_text:
        return True
    try:
        no_docs = not context_docs or len(context_docs) == 0
    except Exception:
        no_docs = False
    if no_docs:
        return True
    a = answer_text.strip()
    very_short = len(a) < 80
    unsure = bool(_UNSURE_RE.search(a))
    looks_like_refusal = bool(re.search(r"\b(can't|cannot|unable)\b.*\b(answer|help)\b", a, re.IGNORECASE))
    return very_short or unsure or looks_like_refusal

def add_disclaimer(text: str) -> str:
    return f"{text}\n\n---\n_{DISCLAIMER_TEXT}_"

def _build_chat_history(messages, turns=8):
    if not messages:
        return []
    chat = [m for m in messages if m["role"] in ("user", "assistant")]
    if chat and chat[-1]["role"] == "user":
        chat = chat[:-1]
    chat = chat[-(2 * turns):]
    out = []
    for m in chat:
        out.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    return out

def parse_structured(text: str):
    data = None
    if not text:
        return {"refusal": False, "final_answer": "", "reasoning": {"intent": "", "plan": "", "recheck": ""}}
    if "```" in text:
        parts = text.split("```")
        for i in range(len(parts)):
            seg = parts[i].strip()
            if seg.startswith("{") and seg.endswith("}"):
                try:
                    data = json.loads(seg); break
                except Exception:
                    pass
            if seg.lower().startswith("json") and i + 1 < len(parts):
                cand = parts[i + 1].strip()
                if cand.startswith("{") and cand.endswith("}"):
                    try:
                        data = json.loads(cand); break
                    except Exception:
                        pass
    if data is None:
        try:
            data = json.loads(text.strip())
        except Exception:
            pass
    if not isinstance(data, dict):
        return {"refusal": False, "final_answer": text.strip(), "reasoning": {"intent": "", "plan": "", "recheck": ""}}
    refusal = bool(data.get("refusal", False))
    final_answer = str(data.get("final_answer", "")).strip()
    reasoning = data.get("reasoning", {}) or {}
    intent = str(reasoning.get("intent", "") or "")
    plan = str(reasoning.get("plan", "") or "")
    recheck = str(reasoning.get("recheck", "") or "")
    return {"refusal": refusal, "final_answer": final_answer, "reasoning": {"intent": intent, "plan": plan, "recheck": recheck}}

def build_search_reason(prompt: str, rag_answer: str, rag_docs) -> tuple[bool, str]:
    reasons, p = [], prompt.lower()
    used = should_force_web_search(prompt) or is_low_confidence_rag(rag_answer, rag_docs)
    triggers = [kw for kw in ["near me", "nearby", "open now", "closest", "nearest", "best", "top", "latest", "update", "2025", "ranking"] if kw in p]
    if should_force_web_search(prompt):
        reasons.append(f"query intent favors live results ({', '.join(repr(t) for t in triggers) if triggers else 'heuristic trigger'})")
    if is_low_confidence_rag(rag_answer, rag_docs):
        reasons.append("RAG low confidence (no/weak context or uncertain text)")
    return used, ("; ".join(reasons) if reasons else "high-confidence RAG answer")

# Small-talk: greetings/help/thanks/check-in (no disclaimer)
def _norm_txt(s: str) -> str:
    s = s.strip().lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9\s'&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_GREET_EXACT = {
    "hi", "hi there", "hello", "hello there", "hey", "hey there",
    "greetings", "good day", "good morning", "good afternoon", "good evening",
    "howdy", "yo",
    "namaste", "namaskar", "vanakkam", "sat sri akal", "adab",
    "salaam", "salam", "salaam alaikum", "salam alaikum", "assalamualaikum", "as-salamu alaykum",
    "bonjour", "hola", "ciao", "ola", "olá", "hallo", "guten tag",
    "konnichiwa", "ni hao", "annyeong", "sawubona", "marhaba", "merhaba",
}
_GREET_PREFIXES = {
    "good morning", "good afternoon", "good evening", "good day",
    "hello", "hi", "hey", "greetings", "howdy",
}
_CHECKIN_PHRASES = {
    "what's up", "whats up", "sup", "wassup", "waddup",
    "how are you", "how r u", "how's it going", "hows it going",
    "how are ya", "how you doing", "how are you doing", "how do you do",
    "how have you been", "hru",
}
_THANKS_SET = {"thanks", "thank you", "thx", "ty", "tysm", "much appreciated"}
_HELP_PATTERNS = [
    r"\bhelp\b", r"what can you do", r"who are you", r"how do i use",
    r"what are your capabilities", r"about you",
]

def detect_smalltalk(q: str) -> str | None:
    nq = _norm_txt(q)
    if nq in _THANKS_SET or any(nq.startswith(t) for t in _THANKS_SET):
        return "thanks"
    if any(re.search(p, nq) for p in _HELP_PATTERNS):
        return "help"
    if nq in _CHECKIN_PHRASES or any(nq.startswith(p) for p in _CHECKIN_PHRASES):
        return "checkin"
    if (nq in _GREET_EXACT) or any(nq.startswith(p) for p in _GREET_PREFIXES) or nq in {"hi medimind", "hello medimind"}:
        return "greet"
    return None

def smalltalk_response(kind: str) -> str:
    if kind == "greet":
        return (
            "Hi! I’m MediMind — a medical information assistant. I can help with medical, health, biology, and pharmacology questions.\n\n"
            "Examples you can try:\n"
            "- I have sharp chest pain that worsens when I lie down — what should I do?\n"
            "- Is doxycycline safe during pregnancy? What are alternatives?\n"
            "- Max adult dose of acetaminophen and liver safety?\n"
            "- Latest (2025) ADA T2D pharmacotherapy updates."
        )
    if kind == "checkin":
        return "All good — ready to help with medical questions. Tell me what you’d like to know about health, medicine, biology, or pharmacology."
    if kind == "help":
        return (
            "I focus strictly on medical, health, biology, and pharmacology topics. Ask me anything in that space.\n\n"
            "Tips:\n"
            "- Include your context (e.g., medications, allergies, conditions) for tailored info.\n"
            "- Add “near me” to find nearby hospitals (I’ll ask for location once).\n"
            "- Add “latest” or a year (e.g., 2025) for guideline updates — I’ll search the web."
        )
    if kind == "thanks":
        return "You’re welcome! If you have another medical question, I’m here to help."
    return ""
# =====================================================================


# ======================== Prompts (structured JSON) ===================
REASONED_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are MediMind, a specialized medical AI. Only answer medical/health/biology queries; politely refuse others.\n"
     "Use ONLY the provided context. If insufficient or ambiguous, ask a short clarifying question.\n"
     "Output MUST be a single JSON object (no extra text) with keys: refusal (bool), final_answer (string), "
     "and reasoning.intent / reasoning.plan / reasoning.recheck (strings). Do NOT include URLs or a 'Sources' section in final_answer."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "Question: {input}"),
    ("system", "Context:\n{context}")
])

WEB_PROMPT = PromptTemplate(
    template=(
        "You are MediMind, a specialized medical AI. Only answer medical/health/biology queries; refuse others.\n"
        "Use the web results to answer. If insufficient, ask a short clarifying question.\n"
        "Output MUST be a single JSON object (no extra text) with keys: refusal (bool), final_answer (string), "
        "and reasoning.intent / reasoning.plan / reasoning.recheck (strings). "
        "Do NOT include URLs or a 'Sources' section in final_answer.\n\n"
        "Question: {question}\n\n"
        "Web results:\n{web_results}\n"
    ),
    input_variables=["question", "web_results"]
)
# =====================================================================


# ============================= Web Search =============================
def web_search_answer(query: str, llm: ChatGroq, k: int = TOP_K):
    depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    include_domains_env = os.getenv("TAVILY_INCLUDE_DOMAINS")
    include_domains = tuple(d.strip() for d in include_domains_env.split(",")) if include_domains_env else tuple()
    try:
        search_resp = tavily_search_cached(query, depth, include_domains, k)
        results = (search_resp or {}).get("results", [])
    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}")
    if not results:
        raise RuntimeError("Tavily returned no results.")

    norm_results = []
    for r in results:
        title = r.get("title") or r.get("url") or "Result"
        link = r.get("url") or ""
        snippet = (r.get("content") or "")[:WEB_SNIPPET_CHARS]
        norm_results.append({"title": title, "link": link, "snippet": snippet})

    results_text = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['snippet']}\nURL: {r['link']}"
        for i, r in enumerate(norm_results)
    )

    msg = llm.invoke(WEB_PROMPT.format(question=query, web_results=results_text))
    parsed = parse_structured(getattr(msg, "content", str(msg)))
    sources_list = [{"title": r["title"], "link": r["link"]} for r in norm_results if r["link"]]
    return parsed, sources_list
# =====================================================================


# ========================== OSM: hospitals ============================
@st.cache_data(ttl=1800, show_spinner=False)
def osm_hospitals_near(lat: float, lon: float, radius_m: int = 6000):
    api = overpy.Overpass()
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});
      way["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});
      relation["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});

      node["healthcare"~"^(hospital|clinic|doctor|emergency_care)$"](around:{radius_m},{lat},{lon});
      way["healthcare"~"^(hospital|clinic|doctor|emergency_care)$"](around:{radius_m},{lat},{lon});
      relation["healthcare"~"^(hospital|clinic|doctor|emergency_care)$"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    result = api.query(query)

    points = []

    # nodes
    for n in result.nodes:
        points.append({
            "name": n.tags.get("name", "Hospital/Clinic"),
            "lat": float(n.lat),
            "lon": float(n.lon),
            "addr": n.tags.get("addr:full") or n.tags.get("addr:street") or "",
            "website": n.tags.get("website") or "",
            "phone": n.tags.get("phone") or n.tags.get("contact:phone") or "",
            "opening_hours": n.tags.get("opening_hours") or ""
        })

    # ways (use center coords if available)
    for w in result.ways:
        clat = getattr(w, "center_lat", None)
        clon = getattr(w, "center_lon", None)
        if clat is None or clon is None:
            if w.nodes:
                avg_lat = sum(float(nd.lat) for nd in w.nodes) / len(w.nodes)
                avg_lon = sum(float(nd.lon) for nd in w.nodes) / len(w.nodes)
                clat, clon = avg_lat, avg_lon
            else:
                continue
        points.append({
            "name": w.tags.get("name", "Hospital/Clinic"),
            "lat": float(clat),
            "lon": float(clon),
            "addr": w.tags.get("addr:full") or w.tags.get("addr:street") or "",
            "website": w.tags.get("website") or "",
            "phone": w.tags.get("phone") or w.tags.get("contact:phone") or "",
            "opening_hours": w.tags.get("opening_hours") or ""
        })

    # relations (center coords)
    for r in result.relations:
        clat = getattr(r, "center_lat", None)
        clon = getattr(r, "center_lon", None)
        if clat is None or clon is None:
            continue
        points.append({
            "name": r.tags.get("name", "Hospital/Clinic"),
            "lat": float(clat),
            "lon": float(clon),
            "addr": r.tags.get("addr:full") or r.tags.get("addr:street") or "",
            "website": r.tags.get("website") or "",
            "phone": r.tags.get("phone") or r.tags.get("contact:phone") or "",
            "opening_hours": r.tags.get("opening_hours") or ""
        })

    # de-dup by name+coords
    seen = set()
    uniq = []
    for p in points:
        key = (p["name"], round(p["lat"], 5), round(p["lon"], 5))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq

def render_osm_map(lat: float, lon: float, points: list, height: int = 520):
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([lat, lon], tooltip="You", icon=folium.Icon(color="blue")).add_to(fmap)
    for p in points:
        d_url = f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={lat:.6f},{lon:.6f};{p['lat']:.6f},{p['lon']:.6f}"
        v_url = f"https://www.openstreetmap.org/?mlat={p['lat']:.6f}&mlon={p['lon']:.6f}#map=16/{p['lat']:.6f}/{p['lon']:.6f}"
        popup = f"<b>{html.escape(p['name'])}</b><br/>{html.escape(p['addr'])}<br/>"
        if p["phone"]:
            popup += f"Phone: {html.escape(p['phone'])}<br/>"
        if p["opening_hours"]:
            popup += f"Hours: {html.escape(p['opening_hours'])}<br/>"
        popup += f"<a href='{d_url}' target='_blank'>Directions</a> · <a href='{v_url}' target='_blank'>Open in OSM</a>"
        folium.Marker([p["lat"], p["lon"]], popup=popup, icon=folium.Icon(color="red", icon="plus")).add_to(fmap)
    st_folium(fmap, width=None, height=height)

def _inject_hover_map_css_once():
    if st.session_state.get("_hover_map_css_injected"):
        return
    st.markdown(
        """
        <style>
        .hosp-list { list-style:none; margin:0; padding:0; }
        .hosp-item { position:relative; display:block; margin:6px 0; }
        .hosp-name { color:#2563eb; text-decoration:none; border-bottom:1px dotted #93c5fd; cursor:pointer; }
        .hosp-card {
            visibility:hidden; opacity:0; transition:opacity .15s ease-in-out;
            position:absolute; left:0; top:125%; z-index:10000; width:min(460px, 90vw);
            background:#0f172a; color:#f8fafc; border-radius:12px; padding:12px;
            box-shadow:0 10px 28px rgba(0,0,0,.30);
        }
        .hosp-item:hover .hosp-card { visibility:visible; opacity:1; }
        .hosp-card-title { font-weight:600; margin-bottom:8px; }
        .hosp-iframe { width:100%; height:240px; border:0; border-radius:8px; }
        .hosp-actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
        .hosp-actions a {
            color:#93c5fd; text-decoration:none; border:1px solid #334155;
            padding:4px 10px; border-radius:999px; font-size:0.9rem;
        }
        .hosp-actions a:hover { background:#1f2937; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_hover_map_css_injected"] = True

def _osm_embed_iframe(lat: float, lon: float, delta: float = 0.01) -> str:
    min_lon, min_lat = lon - delta, lat - delta
    max_lon, max_lat = lon + delta, lat + delta
    src = (
        "https://www.openstreetmap.org/export/embed.html?"
        f"bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}&layer=mapnik&marker={lat:.6f},{lon:.6f}"
    )
    return f'<iframe class="hosp-iframe" loading="lazy" src="{src}"></iframe>'

def render_osm_hover_list(points: list, origin_lat: float | None, origin_lon: float | None, label="Nearby hospitals"):
    if not points:
        return
    _inject_hover_map_css_once()
    items_html = []
    for p in points[:12]:
        name = html.escape(p["name"])
        iframe_html = _osm_embed_iframe(p["lat"], p["lon"], delta=0.008)
        v_url = f"https://www.openstreetmap.org/?mlat={p['lat']:.6f}&mlon={p['lon']:.6f}#map=16/{p['lat']:.6f}/{p['lon']:.6f}"
        d_url = ""
        if origin_lat is not None and origin_lon is not None:
            d_url = f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={origin_lat:.6f},{origin_lon:.6f};{p['lat']:.6f},{p['lon']:.6f}"
        actions = f"<a href='{v_url}' target='_blank'>Open</a>"
        if d_url:
            actions += f" · <a href='{d_url}' target='_blank'>Directions</a>"
        if p.get("website"):
            actions += f" · <a href='{html.escape(p['website'])}' target='_blank'>Website</a>"
        addr_line = html.escape(p.get("addr",""))
        phone_line = f" · {html.escape(p['phone'])}" if p.get("phone") else ""
        item = f"""
        <li class="hosp-item">
          <span class="hosp-name">{name}</span>
          <div class="hosp-card">
            <div class="hosp-card-title">{name}</div>
            <div style="font-size:0.9rem;color:#cbd5e1;">{addr_line}{phone_line}</div>
            {iframe_html}
            <div class="hosp-actions">{actions}</div>
          </div>
        </li>
        """
        items_html.append(item)
    html_block = f"<ul class='hosp-list'>{''.join(items_html)}</ul>"
    with st.popover(label):
        st.markdown(html_block, unsafe_allow_html=True)
# =====================================================================


# ============================ Reasoning/UI ============================
def render_sources_button(sources_list, label="Sources"):
    if not sources_list:
        return
    with st.popover(label):
        st.markdown("Sources")
        for i, s in enumerate(sources_list, 1):
            st.markdown(f"{i}. [{s['title']}]({s['link']})")

def render_reasoning_expander(reasoning: dict):
    if not reasoning:
        return
    with st.expander("🧠 Thinking Process (Intent • Plan • Recheck • Search • Location)"):
        intent = reasoning.get("intent", "").strip()
        plan = reasoning.get("plan", "").strip()
        recheck = reasoning.get("recheck", "").strip()
        search_info = reasoning.get("search", {})
        loc_info = reasoning.get("location", {})
        if intent:
            st.markdown(f"- Clarify user intent: {intent}")
        if plan:
            st.markdown(f"- Output plan: {plan}")
        if recheck:
            st.markdown(f"- Recheck/quality control: {recheck}")
        if search_info:
            tag = "Used web search" if search_info.get("used") else "Did not use web search"
            why = f" — {search_info.get('why','')}" if search_info.get('why') else ""
            st.markdown(f"- Search decision: {tag}{why}")
        if loc_info:
            ltag = "Used location" if loc_info.get("used") else "No location used"
            where = f" — {loc_info.get('where','')}" if loc_info.get("where") else ""
            st.markdown(f"- Location: {ltag}{where}")
# =====================================================================


# ================================ App ================================
def main():
    st.title("Ask Chatbot!")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for m in st.session_state.messages:
        role = m["role"]
        content = m["content"]
        sources = m.get("sources")
        reasoning = m.get("reasoning")
        if role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
                if sources:
                    render_sources_button(sources)
                if reasoning:
                    render_reasoning_expander(reasoning)
        else:
            st.chat_message("user").markdown(content)

    prompt = st.chat_input("Pass your prompt here")
    if not prompt:
        return

    # Echo user and store
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Small talk first (no disclaimer)
        st_kind = detect_smalltalk(prompt)
        if st_kind:
            msg = smalltalk_response(st_kind)
            st.chat_message("assistant").markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        # Non-medical guard (won't block hospital/ER queries)
        if is_clearly_non_medical(prompt):
            refusal = "I cannot answer that question. My function is strictly limited to providing information on medical and health-related topics."
            st.chat_message("assistant").markdown(refusal)
            st.session_state.messages.append({"role": "assistant", "content": refusal})
            return

        vectorstore = get_vectorstore()
        llm = get_llm()

        # Build chat history
        chat_history = _build_chat_history(st.session_state.messages, turns=MEMORY_TURNS)

        # Retriever (MMR for diversity)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT}
        )

        # RAG chain (structured JSON)
        combine_docs_chain = create_stuff_documents_chain(llm, REASONED_QA_PROMPT)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # RAG attempt
        with st.spinner("thinking…"):
            response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        rag_raw = response.get("answer", "")
        rag_docs = response.get("context", [])

        parsed_rag = parse_structured(rag_raw)
        final_answer = parsed_rag.get("final_answer", "").strip()
        refusal = bool(parsed_rag.get("refusal", False))

        # Decide search
        use_web, search_why = build_search_reason(prompt, final_answer, rag_docs)
        is_near_query = any(x in prompt.lower() for x in ["near me", "nearby", "open now", "closest", "nearest"])

        # If "near me" query and we don't have a location yet, ask inline (no sidebar)
        if is_near_query and ("geo" not in st.session_state):
            with st.chat_message("assistant"):
                st.markdown("To find nearby hospitals, share your location:")
                col1, col2 = st.columns([1,1])
                with col1:
                    if get_geolocation and st.button("Use my browser location", key="btn_geo"):
                        g = get_geolocation()
                        if g and "coords" in g and g["coords"]:
                            lat = g["coords"].get("latitude")
                            lon = g["coords"].get("longitude")
                            if lat and lon:
                                st.session_state["geo"] = {"lat": float(lat), "lon": float(lon)}
                                place_txt = reverse_geocode(float(lat), float(lon))
                                st.success(f"Location set: {place_txt}")
                with col2:
                    with st.form("manual_loc"):
                        city = st.text_input("City", key="form_city")
                        region = st.text_input("Region/State", key="form_region")
                        country = st.text_input("Country", key="form_country")
                        ok = st.form_submit_button("Set location")
                    if ok:
                        where = " ".join([t for t in [city, region, country] if t]).strip()
                        if where:
                            coords = geocode_place(where)
                            if coords:
                                st.session_state["geo"] = {"lat": coords[0], "lon": coords[1]}
                                st.success(f"Location set: {where}")
                            else:
                                st.error("Couldn't find that place. Try a more specific city/region.")
                st.info("After setting the location, resend your question (e.g., “24/7 emergency hospitals near me”).")
            st.session_state.messages.append({"role": "assistant", "content": "Please provide your location to proceed."})
            return

        # Continue with normal flow
        if use_web and not refusal:
            with st.chat_message("assistant"):
                status = st.empty()
                status.markdown("searching 🌐...")
                enriched_query = prompt
                # If near-me with location, enrich query
                if is_near_query and ("geo" in st.session_state):
                    lat = st.session_state["geo"]["lat"]; lon = st.session_state["geo"]["lon"]
                    place_txt = reverse_geocode(lat, lon)
                    enriched_query = re.sub(r"\b(near me|nearby)\b", f"near {place_txt}", prompt, flags=re.IGNORECASE)

                parsed_web, sources_list = web_search_answer(enriched_query, llm)

                # Merge reasoning with search + location meta
                reasoning = parsed_web.get("reasoning", {}) or {}
                reasoning["search"] = {"used": True, "why": search_why}
                if "geo" in st.session_state:
                    lat = st.session_state["geo"]["lat"]; lon = st.session_state["geo"]["lon"]
                    reasoning["location"] = {"used": True, "where": reverse_geocode(lat, lon)}
                else:
                    reasoning["location"] = {"used": False, "where": ""}

                if parsed_web.get("refusal"):
                    result_text = "I cannot answer that question. My function is strictly limited to providing information on medical and health-related topics."
                    status.markdown(result_text)
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
                else:
                    out_text = parsed_web["final_answer"].strip()
                    out_text = add_disclaimer(out_text) if not out_text.endswith(DISCLAIMER_TEXT) else out_text
                    status.markdown(out_text)
                    render_sources_button(sources_list)
                    render_reasoning_expander(reasoning)

                    # If near-me and we have a location, show free OSM map + hover list
                    if is_near_query and ("geo" in st.session_state):
                        lat = st.session_state["geo"]["lat"]; lon = st.session_state["geo"]["lon"]
                        with st.spinner("Finding nearby hospitals (OSM)…"):
                            pts = osm_hospitals_near(lat, lon, radius_m=6000)
                        render_osm_map(lat, lon, pts, height=520)
                        render_osm_hover_list(pts, origin_lat=lat, origin_lon=lon, label="Nearby hospitals")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": out_text,
                        "sources": sources_list,
                        "reasoning": reasoning
                    })
        else:
            if refusal:
                result_text = "I cannot answer that question. My function is strictly limited to providing information on medical and health-related topics."
                st.chat_message("assistant").markdown(result_text)
                st.session_state.messages.append({"role": "assistant", "content": result_text})
            else:
                reasoning = parsed_rag.get("reasoning", {}) or {}
                reasoning["search"] = {"used": False, "why": "high-confidence RAG answer"}
                if "geo" in st.session_state:
                    lat = st.session_state["geo"]["lat"]; lon = st.session_state["geo"]["lon"]
                    reasoning["location"] = {"used": True, "where": reverse_geocode(lat, lon)}
                else:
                    reasoning["location"] = {"used": False, "where": ""}

                out_text = final_answer
                out_text = add_disclaimer(out_text) if not out_text.endswith(DISCLAIMER_TEXT) else out_text
                with st.chat_message("assistant"):
                    st.markdown(out_text)
                    render_reasoning_expander(reasoning)

                    # If near-me and we have a location, show free OSM map + hover list
                    if is_near_query and ("geo" in st.session_state):
                        lat = st.session_state["geo"]["lat"]; lon = st.session_state["geo"]["lon"]
                        with st.spinner("Finding nearby hospitals (OSM)…"):
                            pts = osm_hospitals_near(lat, lon, radius_m=6000)
                        render_osm_map(lat, lon, pts, height=520)
                        render_osm_hover_list(pts, origin_lat=lat, origin_lon=lon, label="Nearby hospitals")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": out_text,
                    "reasoning": reasoning
                })

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
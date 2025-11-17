"""
cleaner-ontology.py

JSON-in, JSON-out. Non-destructive. Robust PEM water electrolyzer routing.

Success:
{
  "ok": true,
  "added": <int>,
  "unplaced": ["..."],
  "ontology": <updated root (wrapper preserved)>
}

Error:
{ "ok": false, "error": "...", "details": {...} }

Run:
    python cleaner-ontology.py --ontology /path/Ontology.json --model mistral > sample_output.json
"""

import argparse, json, re, subprocess, sys, difflib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

DEFAULT_ONTOLOGY_PATH = "/Users/hrishilogani/Desktop/soke-test-data/Ontology.json" # Path to input ontology (change it as per your use case)
DEFAULT_OLLAMA_MODEL  = "mistral" # DEFAULT MODEL if not provided in CLI

# --------------------------- Ollama ------------------------------------------

def call_ollama(prompt: str, model: str) -> str:
    res = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return res.stdout.decode("utf-8").strip()

# ----------------------------- I/O -------------------------------------------

def load_ontology(path: Path) -> Tuple[Dict[str, Any], bool]:
    root = json.loads(path.read_text(encoding="utf-8"))
    has_wrapper = isinstance(root, dict) and "data" in root and isinstance(root["data"], dict)
    tree = root["data"] if has_wrapper else root
    if not isinstance(tree, dict):
        raise ValueError("Ontology must be a dict or {'data': dict}.")
    return root, has_wrapper

def clone_root(root: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(root, ensure_ascii=False))

# --------------------------- Merge helpers -----------------------------------

def _merge_list(dst: List[str], src: List[str]) -> int:
    added, seen = 0, {str(x).strip().lower() for x in dst}
    for item in src or []:
        s = str(item).strip()
        if not s:
            continue
        low = s.lower()
        if low not in seen:
            dst.append(s)
            seen.add(low)
            added += 1
    return added

def ensure_dict_list(tree: Dict[str, Any], top: str, sub: str) -> List[str]:
    tree.setdefault(top, {})
    if not isinstance(tree[top], dict):
        tree[top] = {}
    tree[top].setdefault(sub, [])
    if not isinstance(tree[top][sub], list):
        tree[top][sub] = []
    return tree[top][sub]

def ensure_list(tree: Dict[str, Any], top: str) -> List[str]:
    tree.setdefault(top, [])
    if not isinstance(tree[top], list):
        tree[top] = []
    return tree[top]

def merge_additions(tree: Dict[str, Any], additions: Dict[str, Any]) -> int:
    """
    additions example:
      { "Material": {"Alloy": ["brass"]},
        "Elemental Composition": {"Copper": ["Cu"], "Zinc": ["Zn"]} }
    """
    total = 0
    for top, payload in (additions or {}).items():
        if isinstance(payload, dict):
            for sub, terms in payload.items():
                dst = ensure_dict_list(tree, top, sub)
                total += _merge_list(dst, terms if isinstance(terms, list) else [])
        elif isinstance(payload, list):
            dst = ensure_list(tree, top)
            total += _merge_list(dst, payload)
    return total

# ----------------------- Concept extraction ----------------------------------

_STOPWORDS = {"the","a","an","for","under","with","and","or","of","in","on","to","from","by",
              "at","as","into","via","using","overall","conditions","based","using","under"}
_SEP_RE = re.compile(r"[,\;/\|()\[\]{}:+–—\-]+")

def normalize_phrase(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[–—]+", "-", s)          # normalize long dashes to hyphen
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    return s

def heuristic_extract(query: str) -> List[str]:
    q = normalize_phrase(query)
    parts = _SEP_RE.split(q)
    tokens, seen = [], set()
    for p in parts:
        words = [w for w in re.split(r"\s+", p.strip()) if w]
        words = [w for w in words if w.lower() not in _STOPWORDS]
        if not words:
            continue
        # short 1–3-word chunks
        phrase = " ".join(words[:3])
        phrase = normalize_phrase(phrase)
        low = phrase.lower()
        if phrase and low not in seen:
            tokens.append(phrase)
            seen.add(low)
    return tokens

def extract_with_ollama(query: str, full_tree_json: str, model: str) -> List[str]:
    prompt = f"""
Given ONTOLOGY(JSON) and a USER QUERY (short phrase), extract atomic concepts (1–3 words).
Return STRICT JSON: {{ "concepts": ["term1","term2", ...] }}
- lowercase unless chemical formulae (HER, OER, NiFe, IrO2, RuO2)
- no duplicates, no stopwords, no punctuation-only tokens

ONTOLOGY(JSON):
{full_tree_json}

USER QUERY:
{query}
""".strip()
    try:
        raw = call_ollama(prompt, model)
        parsed = json.loads(raw)
        out = [normalize_phrase(str(c)) for c in parsed.get("concepts", []) if str(c).strip()]
        return out
    except Exception:
        return []

def extract_concepts(query: str, full_tree_json: str, model: str) -> List[str]:
    llm = extract_with_ollama(query, full_tree_json, model)
    heur = heuristic_extract(query)
    # union while preserving order preference
    seen, out = set(), []
    for c in llm + heur:
        k = c.lower().strip()
        if k and k not in seen:
            out.append(c)
            seen.add(k)
    return out

# ----------------------------- Knowledge base --------------------------------

# Elements (expanded for PEM)
ELEMENTS = {
    "hydrogen":"H","carbon":"C","nitrogen":"N","oxygen":"O","fluorine":"F",
    "sodium":"Na","magnesium":"Mg","aluminum":"Al","silicon":"Si","phosphorus":"P","sulfur":"S",
    "chlorine":"Cl","potassium":"K","calcium":"Ca","titanium":"Ti","vanadium":"V","chromium":"Cr",
    "manganese":"Mn","iron":"Fe","cobalt":"Co","nickel":"Ni","copper":"Cu","zinc":"Zn",
    "molybdenum":"Mo","palladium":"Pd","silver":"Ag","tin":"Sn","tungsten":"W",
    "platinum":"Pt","gold":"Au","lead":"Pb","ruthenium":"Ru","iridium":"Ir","rhodium":"Rh"
}
SYMBOL_TO_NAME = {v.lower(): k for k, v in ELEMENTS.items()}

# Common alloys (carry over & extend)
COMMON_ALLOYS: Dict[str, Dict[str, List[str]]] = {
    "brass":          {"synonyms": ["brass"],                                  "elements": ["copper","zinc"]},
    "bronze":         {"synonyms": ["bronze","phosphor bronze"],             "elements": ["copper","tin"]},
    "stainless steel":{"synonyms": ["stainless steel","ss","304","316"],  "elements": ["iron","chromium","nickel","molybdenum"]},
    "steel":          {"synonyms": ["steel","carbon steel","mild steel"],   "elements": ["iron","carbon"]},
    "cupronickel":    {"synonyms": ["cupronickel","cu-ni","cu ni","cuni"],"elements": ["copper","nickel"]},
    "monel":          {"synonyms": ["monel"],                                  "elements": ["nickel","copper"]},
    "inconel":        {"synonyms": ["inconel"],                                "elements": ["nickel","chromium"]},
    "hastelloy":      {"synonyms": ["hastelloy"],                              "elements": ["nickel","molybdenum","chromium"]},
    "nichrome":       {"synonyms": ["nichrome","ni-cr","ni cr"],            "elements": ["nickel","chromium"]},
    "duralumin":      {"synonyms": ["duralumin","duraluminium"],             "elements": ["aluminum","copper","magnesium","manganese"]},
    "alnico":         {"synonyms": ["alnico"],                                 "elements": ["aluminum","nickel","cobalt","iron"]},
    "nickel-iron":    {"synonyms": ["nickel-iron","ni-fe","nife"],          "elements": ["nickel","iron"]},
}

# PEM-specific canonical concepts (synonyms -> additions). Each entry
# routes to one or more ontology placements and canonical tokens.
KB_ENTRIES: List[Dict[str, Any]] = [
    # Application / system
    {"synonyms": ["pem electrolyzer","pem water electrolyzer","pem electrolysis","pemwe",
                   "proton exchange membrane electrolyzer","proton exchange membrane water electrolysis"],
     "additions": {"Application": {"PEM Electrolyzer": ["PEM electrolyzer"]}}},

    # Membrane & ionomer
    {"synonyms": ["proton exchange membrane","pfsa membrane","pfsa","ionomer",
                   "nafion","nafion 117","aquivion","flemion"],
     "additions": {"Material": {"Membranes": ["PFSA membrane"]}}},

    {"synonyms": ["membrane electrode assembly","mea"],
     "additions": {"Material": {"Membrane Electrode Assembly (MEA)": ["MEA"]}}},

    # GDL / PTL / BPP
    {"synonyms": ["gas diffusion layer","gdl","carbon paper","carbon cloth","toray paper","sigracet"],
     "additions": {"Material": {"Gas Diffusion Layer (GDL)": ["gas diffusion layer"]}}},

    {"synonyms": ["porous transport layer","ptl","titanium felt","titanium mesh"],
     "additions": {"Material": {"Porous Transport Layer (PTL)": ["porous transport layer"]},
                    "Elemental Composition": {"Titanium": ["Ti"]}}},

    {"synonyms": ["bipolar plate","bpp"],
     "additions": {"Material": {"Bipolar Plate": ["bipolar plate"]}}},

    # Catalysts (OER / HER)
    {"synonyms": ["iridium dioxide","iro2","ir o2","iro₂","iridium oxide"],
     "additions": {"Material": {"Metal Oxides": ["IrO2"]},
                    "Elemental Composition": {"Iridium": ["Ir"]},
                    "Reaction": {"Oxygen Evolution Reaction (OER)": ["OER"]}}},

    {"synonyms": ["ruthenium dioxide","ruo2","ru o2","ruo₂","ruthenium oxide"],
     "additions": {"Material": {"Metal Oxides": ["RuO2"]},
                    "Elemental Composition": {"Ruthenium": ["Ru"]},
                    "Reaction": {"Oxygen Evolution Reaction (OER)": ["OER"]}}},

    {"synonyms": ["platinum on carbon","pt/c","pt on c","ptc","platinum black","platinum catalyst"],
     "additions": {"Elemental Composition": {"Platinum": ["Pt"]},
                    "Reaction": {"Hydrogen Evolution Reaction (HER)": ["HER"]}}},

    # Measurement / protocols
    {"synonyms": ["electrochemical impedance spectroscopy","eis","nyquist"],
     "additions": {"Performance & Stability": {"EIS": ["electrochemical impedance spectroscopy"]}}},

    {"synonyms": ["linear sweep voltammetry","lsv","polarization curve"],
     "additions": {"Performance & Stability": {"Polarization": ["linear sweep voltammetry"]}}},

    {"synonyms": ["chronoamperometry","ca","chronopotentiometry","cp","durability test"],
     "additions": {"Performance & Stability": {"Stability": ["chronoamperometry"]}}},

    {"synonyms": ["faradaic efficiency","fe%","faradaic"],
     "additions": {"Performance & Stability": {"Faradaic Efficiency": ["faradaic efficiency"]}}},

    {"synonyms": ["ir drop","iR drop","IR-corrected","uncompensated resistance"],
     "additions": {"Performance & Stability": {"IR Drop": ["iR drop"]}}},
]

# Precompile KB regexes for efficiency
KB_REGEX: List[Tuple[Dict[str, Any], List[re.Pattern]]] = []
for entry in KB_ENTRIES:
    pats = [re.compile(rf"\\b{re.escape(s)}\\b", re.I) for s in entry["synonyms"]]
    KB_REGEX.append((entry, pats))

# Alloy synonym regex cache
ALLOY_SYNONYM_RES = [(canon, [re.compile(rf"\\b{re.escape(s)}\\b", re.I) for s in data["synonyms"]]) for canon, data in COMMON_ALLOYS.items()]

# Element-pair alloys (name or symbol)
PAIR_ALLOY_RES = [
    re.compile(r"\b(ni|nickel)[\s\-–—]?(fe|iron)\b", re.I),
    re.compile(r"\b(cu|copper)[\s\-–—]?(zn|zinc)\b", re.I),
    re.compile(r"\b(ni|nickel)[\s\-–—]?(cr|chromium)\b", re.I),
    re.compile(r"\b(cu|copper)[\s\-–—]?(ni|nickel)\b", re.I),
]

# ----------------------------- Routing utils ---------------------------------

JUNK_RES = [
    re.compile(r"^[<>]=?$"),              # stray comparators
    re.compile(r"^pH[<>]=?$", re.I),      # pH>, pH<, etc.
    re.compile(r"^[\W_]+$"),             # punctuation-only
    re.compile(r"^\d+(\.\d+)?\s*°?C$"),# bare temperatures
]

def is_junk(token: str) -> bool:
    t = token.strip()
    if not t or len(t) <= 1:
        return True
    return any(r.search(t) for r in JUNK_RES)

# General domain patterns
LDH_RE   = re.compile(r"\blayered double hydroxides?\b|\bLDH\b", re.I)
WS_RE    = re.compile(r"\boverall water splitting\b|\bwater splitting\b|\belectrolysis of water\b", re.I)
ALK_RE   = re.compile(r"\balkaline\b|\bbasic\b|\bpH\s*[>≥]\s*7(\.0+)?\b|\bpH\s*=\s*(8|9|10|11|12|13|14)\b", re.I)
ACID_RE  = re.compile(r"\bacidic\b|\bpH\s*[<≤]\s*7(\.0+)?\b|\bpH\s*=\s*([1-6])\b", re.I)
NEUT_RE  = re.compile(r"\bneutral\b|\bpH\s*=?\s*7\b", re.I)
OER_RE   = re.compile(r"\bOER\b|\boxygen evolution\b", re.I)
HER_RE   = re.compile(r"\bHER\b|\bhydrogen evolution\b", re.I)

# Formula-like catalysts: IrO2, RuO2, Pt/C etc.
IRIDIA_RE = re.compile(r"\bIrO2\b|\biridium dioxide\b|\biridium oxide\b", re.I)
RUTHEN_RE = re.compile(r"\bRuO2\b|\bruthenium dioxide\b|\bruthenium oxide\b", re.I)
PTC_RE    = re.compile(r"\bPt\s*/\s*C\b|\bPt/C\b|\bplatinum on carbon\b|\bplatinum black\b", re.I)

# ----------------------------- Routing primitives ----------------------------

def additions_for_alloy(name: str, element_names: List[str]) -> Dict[str, Any]:
    additions: Dict[str, Any] = {
        "Material": {"Alloy": [name]}
    }
    if element_names:
        comp: Dict[str, List[str]] = {}
        for ename in element_names:
            bucket = ename.capitalize()
            symbol = ELEMENTS.get(ename, ename[:2]).upper()
            comp.setdefault(bucket, [])
            comp[bucket].append(symbol)
        additions["Elemental Composition"] = comp
    return additions

# ----------------------------- Knowledge routing -----------------------------

def route_by_kb(text: str) -> List[Dict[str, Any]]:
    """Match PEM KB entries in the full text (exact word-boundary), fallback fuzzy."""
    matches: List[Dict[str, Any]] = []
    low = text.lower()

    # Exact word-bound matches
    for entry, regexes in KB_REGEX:
        if any(r.search(text) for r in regexes):
            matches.append(entry["additions"])

    if matches:
        return matches

    # Fallback: fuzzy over all synonyms (useful for slight typos)
    synonyms = []
    back = {}
    for entry in KB_ENTRIES:
        for s in entry["synonyms"]:
            synonyms.append(s)
            back[s.lower()] = entry["additions"]
    guess = difflib.get_close_matches(low, [s.lower() for s in synonyms], n=2, cutoff=0.88)
    for g in guess:
        matches.append(back[g])
    return matches

# Alloys (KB + pairs)
ALLOY_SYMS = {v.lower(): k for k, v in ELEMENTS.items()}

def route_alloy_by_kb(text: str) -> Tuple[str, List[str]] | None:
    low = text.lower()
    for canon, res_list in ALLOY_SYNONYM_RES:
        if any(r.search(text) for r in res_list):
            return canon, COMMON_ALLOYS[canon]["elements"]
    # fuzzy against canonical names & synonyms
    all_syns = []
    syn_to_canon = {}
    for canon, data in COMMON_ALLOYS.items():
        for s in [canon] + data["synonyms"]:
            all_syns.append(s)
            syn_to_canon[s.lower()] = canon
    close = difflib.get_close_matches(low, [s.lower() for s in all_syns], n=1, cutoff=0.86)
    if close:
        canon = syn_to_canon[close[0]]
        return canon, COMMON_ALLOYS[canon]["elements"]
    return None

def route_alloy_by_pair(text: str) -> Tuple[str, List[str]] | None:
    for r in PAIR_ALLOY_RES:
        m = r.search(text)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        a_name = SYMBOL_TO_NAME.get(a.lower(), a.lower())
        b_name = SYMBOL_TO_NAME.get(b.lower(), b.lower())
        pair = {frozenset({"nickel","iron"}): ("nickel-iron", ["nickel","iron"]),
                frozenset({"copper","zinc"}): ("brass", ["copper","zinc"]),
                frozenset({"nickel","chromium"}): ("nichrome", ["nickel","chromium"]),
                frozenset({"copper","nickel"}): ("cupronickel", ["copper","nickel"])}
        key = frozenset({a_name, b_name})
        if key in pair:
            return pair[key]
        return f"{a_name}-{b_name}", [a_name, b_name]
    # Compact symbols like NiFe, CuZn
    m = re.search(r"\b([A-Z][a-z]?)([A-Z][a-z]?)\b", text)
    if m:
        a_sym, b_sym = m.group(1).lower(), m.group(2).lower()
        if a_sym in SYMBOL_TO_NAME and b_sym in SYMBOL_TO_NAME:
            a_name, b_name = SYMBOL_TO_NAME[a_sym], SYMBOL_TO_NAME[b_sym]
            return route_alloy_by_pair(f"{a_name}-{b_name}")
    return None

# ----------------------------- Domain routing --------------------------------

def route_text_level(text: str) -> List[Dict[str, Any]]:
    """Route using the full phrase (captures multi-word synonyms)."""
    adds: List[Dict[str, Any]] = []

    # PEM KB entries
    adds.extend(route_by_kb(text))

    # Alloys
    kb_alloy = route_alloy_by_kb(text)
    if kb_alloy:
        name, elems = kb_alloy
        adds.append(additions_for_alloy(name, elems))
    else:
        pair = route_alloy_by_pair(text.replace("–","-").replace("—","-"))
        if pair:
            name, elems = pair
            adds.append(additions_for_alloy(name, elems))

    # Layered Double Hydroxides
    if LDH_RE.search(text):
        adds.append({"Material": {"Layered Double Hydroxides": ["layered double hydroxides"]}})

    # Processes / environment / reactions
    if WS_RE.search(text):
        adds.append({"Process": {"Water Electrolysis": ["water splitting"]}})
    if ALK_RE.search(text):
        adds.append({"Environment": {"Alkaline": ["alkaline"]}})
    if ACID_RE.search(text):
        adds.append({"Environment": {"Acidic": ["acidic"]}})
    if NEUT_RE.search(text):
        adds.append({"Environment": {"Neutral": ["neutral pH"]}})

    # Catalysts by formula-like patterns
    if IRIDIA_RE.search(text):
        adds.append({"Material": {"Metal Oxides": ["IrO2"]}})
        adds.append({"Elemental Composition": {"Iridium": ["Ir"]}})
        adds.append({"Reaction": {"Oxygen Evolution Reaction (OER)": ["OER"]}})
    if RUTHEN_RE.search(text):
        adds.append({"Material": {"Metal Oxides": ["RuO2"]}})
        adds.append({"Elemental Composition": {"Ruthenium": ["Ru"]}})
        adds.append({"Reaction": {"Oxygen Evolution Reaction (OER)": ["OER"]}})
    if PTC_RE.search(text):
        adds.append({"Elemental Composition": {"Platinum": ["Pt"]}})
        adds.append({"Reaction": {"Hydrogen Evolution Reaction (HER)": ["HER"]}})

    return adds


def route_concept(concept: str) -> List[Dict[str, Any]]:
    """Route a short concept (1–3 words). May return multiple addition dicts."""
    adds: List[Dict[str, Any]] = []
    c = concept.strip()
    if is_junk(c):
        return adds
    text = c if any(ch.isupper() for ch in c) else c.lower()

    # Try text-level router on the concept itself
    adds.extend(route_text_level(text))

    # Single elements
    w = text.strip().lower()
    if w in ELEMENTS:
        adds.append({"Elemental Composition": {w.capitalize(): [ELEMENTS[w]]}})
    if w in SYMBOL_TO_NAME:
        name = SYMBOL_TO_NAME[w]
        adds.append({"Elemental Composition": {name.capitalize(): [w.upper()]}})

    # HER/OER keywords
    if HER_RE.search(text):
        adds.append({"Reaction": {"Hydrogen Evolution Reaction (HER)": ["HER"]}})
    if OER_RE.search(text):
        adds.append({"Reaction": {"Oxygen Evolution Reaction (OER)": ["OER"]}})

    return adds

# ----------------------------- Main runner -----------------------------------

def run(ontology_path: Path, model: str) -> Dict[str, Any]:
    try:
        if not ontology_path.exists():
            return {"ok": False, "error": f"Ontology not found: {ontology_path}"}

        root, has_wrapper = load_ontology(ontology_path)
        updated = clone_root(root)
        tree = updated["data"] if has_wrapper else updated

        # seeds (read-only)
        user_provided: List[str] = []
        kw = tree.get("Keyword")
        if isinstance(kw, dict):
            user_provided = (kw.get("UserProvided", []) or [])
        elif isinstance(kw, list):
            user_provided = kw

        if not user_provided:
            return {"ok": True, "added": 0, "unplaced": [], "notice": "No user keywords found.", "ontology": updated}

        full_tree_json = json.dumps(tree, ensure_ascii=False)
        total_added = 0
        unplaced: List[str] = []

        for raw in user_provided:
            phrase = str(raw).strip()
            if not phrase:
                continue

            # 1) Route at phrase-level first (captures multiword synonyms)
            for add in route_text_level(phrase):
                total_added += merge_additions(tree, add)

            # 2) Extract short concepts then route them
            concepts = extract_concepts(phrase, full_tree_json, model)
            for concept in concepts:
                adds = route_concept(concept)
                if adds:
                    for add in adds:
                        total_added += merge_additions(tree, add)
                else:
                    unplaced.append(concept)

        # dedupe unplaced
        unplaced = sorted(list(dict.fromkeys([u for u in unplaced if not is_junk(u)])))

        return {"ok": True, "added": total_added, "unplaced": unplaced, "ontology": updated}

    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "error": "Ollama call failed",
            "details": {
                "returncode": e.returncode,
                "stdout": (e.stdout or b"").decode("utf-8", errors="replace"),
                "stderr": (e.stderr or b"").decode("utf-8", errors="replace"),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e.__class__.__name__), "details": {"message": str(e)}}


def main():
    ap = argparse.ArgumentParser(description="Ontology updater for PEM water electrolysis (JSON stdout).")
    ap.add_argument("--ontology", "-o", default=DEFAULT_ONTOLOGY_PATH)
    ap.add_argument("--model", "-m", default=DEFAULT_OLLAMA_MODEL)
    args = ap.parse_args()

    result = run(Path(args.ontology), args.model)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
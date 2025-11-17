`cleaner-ontology.py` reads your ontology JSON, looks at `Keyword.UserProvided` phrases, and **enriches** the ontology by placing canonical terms in the right buckets (Materials, Reactions, Application, Performance, Environment, etc.).
It’s **non-destructive**: it never edits your file; it **always prints JSON** to `stdout`, so you can do:

```bash
python cleaner-ontology.py --ontology Ontology.json --model mistral > output.json
```

---

## Main Workflow (how we use it)

1. **Collect user queries** into the ontology under:
   (this has been implemented in the existing workflow, so user's query will automatically be written to our initial ontology file when they use this tool through streamlit)

```json
"Keyword": { "UserProvided": ["...free-text terms..."] }
```

2. **Run this updater** to normalize and categorize those terms.
   It uses PEM-specific knowledge (membranes/ionomers, MEA/GDL/PTL/BPP, IrO₂/RuO₂/PT/C, HER/OER, EIS/LSV/CA/CP, etc.), common alloys (brass, bronze, stainless, nichrome, cupronickel…), element-pair alloys (Ni–Fe, Cu–Zn…), fuzzy matching, and junk filtering.

3. **Capture the JSON output** and store or diff it as you like:

```bash
python cleaner-ontology.py --ontology Ontology.json > output.json
```

---

## CLI Usage & Flags

```bash
python cleaner-ontology.py [--ontology PATH] [--model NAME]
```

* `--ontology`, `-o`
  Path to the ontology JSON file to read.
  **Default:** `whatever/path/you/specify/here/Ontology.json`
  **Tip:** use absolute paths for CI.

* `--model`, `-m`
  Name of the Ollama model to use for concept extraction (e.g., `mistral`, `llama3`).
  **Default:** `mistral`
  The script is robust even if the LLM fails; it falls back to heuristics/KB.

### Examples

```bash
# Typical run
python cleaner-ontology.py --ontology ./Ontology.json --model mistral > output.json

# Use a custom model build with larger context
python cleaner-ontology.py -o ./Ontology.json -m mistral-32k > output.json

# Quick check (pretty-print to console)
python cleaner-ontology.py -o ./Ontology.json | jq .
```

---

## Output Contract (always JSON)

On success:

```json
{
  "ok": true,
  "added": 12,
  "unplaced": ["...concepts we skipped..."],
  "ontology": { "data_or_root_here": "..." }
}
```

On error (still valid JSON):

```json
{
  "ok": false,
  "error": "Ollama call failed",
  "details": { "returncode": 127, "stdout": "", "stderr": "..." }
}
```

> The script **never** overwrites your input file. You decide where to save the output (`> output.json`).

---

## Changing Defaults (if you don’t want to pass flags)

Inside the script near the top:

```python
DEFAULT_ONTOLOGY_PATH = "/Users/hrishilogani/Desktop/soke-test-data/Ontology.json"
DEFAULT_OLLAMA_MODEL  = "mistral"
```

Change those two lines to your preferred defaults (e.g., repo-relative path):

```python
DEFAULT_ONTOLOGY_PATH = "./Ontology.json"
DEFAULT_OLLAMA_MODEL  = "mistral-32k"
```

Now you can simply run:

```bash
python ontology-cleaner.py > output.json
```

*(Optional)* If you’d rather use environment variables, you can wrap the script with a tiny launcher or modify it to read `os.getenv("ONTOLOGY_PATH")` / `os.getenv("OLLAMA_MODEL")` before falling back to the hard-coded defaults.

---

## What Gets Recognized (high level)

* **PEM System:** “PEM electrolyzer”, “PEMWE”, “proton exchange membrane electrolyzer” → `Application → PEM Electrolyzer`.
* **Membranes/Ionomers:** “Nafion (117, 211, …)”, “Aquivion”, “PFSA” → `Material → Membranes`.
* **MEA / GDL / PTL / BPP:** “MEA”, “gas diffusion layer (GDL)”, “titanium PTL”, “bipolar plate”.
* **Catalysts:** “IrO₂ / RuO₂” (OER), “Pt/C” (HER), NiFe LDH → places under `Material` + `Elemental Composition` + `Reaction`.
* **Reactions:** HER, OER, water splitting.
* **Performance:** overpotential, polarization (LSV), EIS, chronoamperometry/chronopotentiometry, Faradaic efficiency, iR drop.
* **Environment:** Alkaline / Acidic / Neutral, temperatures, etc.
* **Alloys:** brass, bronze, stainless steel, nichrome, cupronickel, Ni–Fe, Cu–Zn, Ni–Cr… with fuzzy matching and symbol/name pair detection.

Anything uncertain is **kept out of the ontology** and listed in `"unplaced"` so you can review it.

---

## Sample Queries to Test

1. **Catalyst + reactions**

```
"Bifunctional IrO₂ anode and Pt/C cathode for PEM water electrolysis with low overpotential"
```

2. **Stack hardware**

```
"PEM electrolyzer stack with Nafion 117 membrane, titanium porous transport layers, and bipolar plates under 80°C"
```

Run:

```bash
python ontology-cleaner.py -o Ontology.json -m mistral > output.json
```

---

## Integration Tips

* **Diffing:**
  Compare `output.json["ontology"]` with your baseline to see what changed:

  ```bash
  jq '.ontology' output.json > updated.json
  diff -u Ontology.json updated.json
  ```

* **Pipelines:**
  Keep the upstream file immutable, and store `output.json` as an artifact.
  If you want to “accept” changes, you can replace the canonical ontology with the updated one after review.

* **Parsing:**
  Use `jq` or a JSON library to inspect `"added"` and `"unplaced"` to drive dashboards or QA checks.

---

## Troubleshooting

* **Ollama not found / model missing**
  You’ll get `{ "ok": false, "error": "Ollama call failed", "details": ... }`.
  Fix: `ollama pull mistral` (or your chosen tag), or set `--model` appropriately.
  The script still uses heuristics/KB, so most routing works even if the LLM fails.

* **Too-large prompts (very big ontology)**
  If you use a small-context model, the script still relies on heuristics and the PEM KB, so results are stable. For larger context, create a higher-context model in Ollama (e.g., `num_ctx 32768`) and run with `--model mistral-32k`.

* **Weird tokens like `pH>`**
  These are filtered by a junk detector; they won’t be inserted into the ontology.

---

## Design Notes (why it’s robust)

* **Two passes:** phrase-level routing (captures multiword synonyms) + concept-level extraction.
* **PEM Knowledge Base:** curated synonyms mapped to canonical placements.
* **Alloy KB + pairs:** handles common names *and* element-pair patterns (Ni–Fe, Cu–Zn…).
* **Fuzzy match:** tolerates small typos (“stainles steel” → “stainless steel”).
* **Multi-placement:** e.g., IrO₂ → `Material`, `Elemental Composition`, and `OER`.
* **Non-destructive:** always JSON to stdout; you choose when and where to persist.

---

Note: you can look at the sample output here: `ontology-cleaner/sample-output.json` and sample_input here: `ontology-cleaner/sample-input.json`.

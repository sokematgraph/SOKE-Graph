# Ontology Generator (via Ollama LLM)

This CLI tool generates a **domain-specific ontology** automatically using a local Ollama LLM (e.g. Mistral).
It uses a reference `example.json` file as a **format guide** (not as content), and adapts that structure to any domain you specify.

---

## Features

- Generate ontology JSONs for *any domain* (e.g. "Battery Management Systems", "Healthcare Data", "Robotics").
- Automatically starts **Ollama** if it's not running.
- Automatically pulls the model (default: `mistral`).
- Embeds `example.json` as a structural reference.
- Ensures **strictly valid JSON** output (auto-repairs if needed).
- Simple CLI options and reproducible prompt format.

---

## Requirements

- [Ollama](https://ollama.com/download) installed and available on your `PATH`.
- Python 3.8+.
- A local model like `mistral` or `llama3` pulled into Ollama.

---

## Usage

### Basic Example

```bash
python ontology_gen.py --domain "battery management systems"
```

This will:

- Ensure `ollama serve` is running
- Pull the default `mistral` model if needed
- Use a short excerpt from `example.json` as a guide
- Print the generated ontology JSON to your terminal

### Save Output to a File

```bash
python ontology_gen.py --domain "fuel cells" --out fuelcells_ontology.json
```

### Use a Different Model

```bash
python ontology_gen.py --domain "biomedical imaging" --model llama3
```

### Include the Full Example JSON

By default, only a **short excerpt** (2 subkeys per section) from `example.json` is included in the prompt for speed.

If you want to use the **full** example file instead:

```bash
python ontology_gen.py --domain "hydrogen storage materials" --include-full-example
```

### Use a Custom Example File

```bash
python ontology_gen.py --domain "supply chain logistics" --example /path/to/your/example.json
```

### Print the Generated Prompt (for debugging)

```bash
python ontology_gen.py --domain "artificial intelligence" --print-prompt
```

---

## Prompt Logic

The model sees a prompt like this:

```
You are an ontology expert in "battery management systems".
Your job is to design a compact, high-signal ontology for this domain.

Use the JSON format and style demonstrated by the reference example (keys → subkeys → list of synonyms/aliases).
Do NOT copy the content—adapt it to the new domain.

Constraints:
- Output must be strictly valid JSON
- 5–10 top-level sections
- Each section contains key: [list of aliases]
- No markdown or extra text
```

Then your `example.json` is appended as a **format reference** — not as data to copy.

---

## Notes

- **`ollama serve` vs `ollama run`**: This script uses the REST API (`/api/chat`), so it needs the Ollama server running. It will automatically start `ollama serve` in the background if it's not detected.

- **Example file location**: By default, it looks for `example.json` in the **same directory** as the script. You can override this using `--example`.

---

## Example Output

For domain: `battery management systems`, you might get something like:

```json
{
  "Components": {
    "Battery Cell": ["Li-ion Cell", "Energy Storage Unit"],
    "BMS Controller": ["Battery Control Unit", "BMS Board"]
  },
  "Safety Mechanisms": {
    "Thermal Management": ["Cooling System", "Heat Regulation"],
    "Fault Detection": ["Error Monitoring", "Diagnostics"]
  },
  "Performance Metrics": {
    "Efficiency": ["Energy Conversion Rate", "Power Output"],
    "Lifetime": ["Cycle Life", "Durability"]
  }
}
```

---

## Example CLI Commands

| Task | Command |
|------|---------|
| Generate ontology for AI models | `python ontology_gen.py --domain "AI model architectures"` |
| Use custom example file | `python ontology_gen.py --domain "Fuel Cells" --example ./reference.json` |
| Force full example | `python ontology_gen.py --domain "Nanomaterials" --include-full-example` |
| Output to file | `python ontology_gen.py --domain "Robotics" --out robotics.json` |
# Ontology Generator (JSON-LD via Ollama LLM)

This CLI tool generates a **domain-specific ontology JSON-LD** automatically using a local Ollama LLM (e.g. Mistral).
It uses `examples/ontology-refactored.jsonld` as a **format guide** and adapts it to any domain.

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
python ontology-gen.py --domain "battery management systems"
```

This will:

- Ensure `ollama serve` is running
- Pull the default `mistral` model if needed
- Use a short excerpt from reference JSON-LD as a guide
- Print the generated ontology JSON to your terminal

### Save Output to a File

```bash
python ontology-gen.py --domain "fuel cells" --out fuelcells_ontology.jsonld
```

### Use a Different Model

```bash
python ontology-gen.py --domain "biomedical imaging" --model llama3
```

### Include the Full Example JSON

By default, only a **short excerpt** from the reference JSON-LD is included in the prompt for speed.

If you want to use the **full** example file instead:

```bash
python ontology-gen.py --domain "hydrogen storage materials" --include-full-example
```

### Use a Custom Example File

```bash
python ontology-gen.py --domain "supply chain logistics" --example /path/to/your/ontology.jsonld
```

### Print the Generated Prompt (for debugging)

```bash
python ontology-gen.py --domain "artificial intelligence" --print-prompt
```

---

## Prompt Logic

The model sees a prompt like this:

```
You are an ontology expert in "battery management systems".
Generate a strict JSON-LD ontology with keys @context and @graph.
Each @graph node contains @id, @type, skos:prefLabel, skos:altLabel.
No markdown or extra text.
```

Then a reference JSON-LD is appended as a **format reference** — not as data to copy.

---

## Notes

- **`ollama serve` vs `ollama run`**: This script uses the REST API (`/api/chat`), so it needs the Ollama server running. It will automatically start `ollama serve` in the background if it's not detected.

- **Example file location**: By default, it looks for `example.json` in the **same directory** as the script. You can override this using `--example`.

---

## Example Output (JSON-LD)

For domain: `battery management systems`, you might get something like:

```json
{
  "@context": {
    "domain": "https://example.org/domain#",
    "skos": "http://www.w3.org/2004/02/skos/core#"
  },
  "@graph": [
    {
      "@id": "domain:BatteryManagementSystem",
      "@type": "Device",
      "skos:prefLabel": "Battery Management System",
      "skos:altLabel": ["BMS", "battery controller"]
    }
  ]
}
```

---

## Example CLI Commands

| Task                            | Command                                                                     |
| ------------------------------- | --------------------------------------------------------------------------- |
| Generate ontology for AI models | `python ontology-gen.py --domain "AI model architectures"`                  |
| Use custom example file         | `python ontology-gen.py --domain "Fuel Cells" --example ./reference.jsonld` |
| Force full example              | `python ontology-gen.py --domain "Nanomaterials" --include-full-example`    |
| Output to file                  | `python ontology-gen.py --domain "Robotics" --out robotics.jsonld`          |
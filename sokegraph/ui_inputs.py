"""ui_agent.py – Colab‑friendly ipywidgets UI for SOKEGraph

Key change ✨
------------
• The *API Keys* chooser is now **shown only** when the selected AI agent is
  one that requires keys (OpenAI, Gemini, or Llama). Ollama does not display
  or validate an API‑key file.

Drop‑in replacement: nothing else in your pipeline needs to change.
"""

import os
from types import SimpleNamespace

import ipywidgets as widgets
from IPython.display import clear_output, display
from ipyfilechooser import FileChooser


class SOKEGraphUI:
    """Interactive widget UI for configuring a SOKEGraph run inside Colab."""

    # ---------------------------------------------------------------------
    # Helper ----------------------------------------------------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def _needs_api_key(ai_name: str) -> bool:
        """Return *True* if the chosen AI backend requires an API‑key file."""
        return ai_name.lower() in {"openai", "gemini", "llama"}

    # ---------------------------------------------------------------------
    # Init ------------------------------------------------------------------
    # ---------------------------------------------------------------------
    def __init__(self):
        # --------------------------------------------------- parameters holder
        self.params = SimpleNamespace(
            input_type="",
            AI="",
            kg_type="",
            number_papers=None,
            paper_query_file="",
            pdf_zip="",
            keywords_file="",
            ontology_file="",
            api_keys_file="",
            kg_credentials_file="",
            output_dir="",
        )

        # ----------------------------------------------------------- widgets
        self.input_type_selector = widgets.Dropdown(
            options=["Select Option", "Number of Papers", "PDF Zip File"],
            description="Paper Input Type:",
        )

        self.ai_selector = widgets.Dropdown(
            options=["openAI", "gemini", "llama", "ollama"],
            description="AI Agent:",
        )

        self.kg_selector = widgets.Dropdown(
            options=["neo4j"],
            description="KG Type:",
        )

        self.number_papers_input = widgets.IntText(description="Number of Papers:")

        # ------------------- File choosers (lazy: default hidden until used)
        def _make_fc(title, pattern, dirs=False):
            fc = FileChooser()
            fc.title = title
            fc.filter_pattern = pattern
            fc.show_only_dirs = dirs
            fc.use_dir_icons = dirs
            return fc

        self.paper_query_chooser = _make_fc("📄 Select Query File (.txt)", "*.txt")
        self.pdf_zip_chooser = _make_fc("🗂 Select ZIP File of PDFs", "*.zip")
        self.keywords_chooser = _make_fc("📝 Select Keywords File", "*.txt")
        self.ontology_chooser = _make_fc("📖 Select Ontology File", "*.json")
        self.api_keys_chooser = _make_fc("🔐 Select API Keys File", "*.txt")
        self.credentials_chooser = _make_fc("🔐 Select KG Credentials File", "*.json")
        self.output_dir_chooser = _make_fc("📁 Select Output Directory", "*", dirs=True)

        # ------------------------------------------------------ other widgets
        self.submit_button = widgets.Button(description="Submit ✅")
        self.output = widgets.Output()
        self.conditional_inputs = widgets.VBox([])

        # ---------------------------------------------------- event bindings
        self.input_type_selector.observe(self._update_inputs, names="value")
        self.ai_selector.observe(self._update_inputs, names="value")
        self.submit_button.on_click(self._on_submit_clicked)

    # ---------------------------------------------------------------------
    # Dynamic UI composition ------------------------------------------------
    # ---------------------------------------------------------------------
    def _update_inputs(self, change=None):  # noqa: D401 – simple name is fine.
        with self.output:
            clear_output()

        children = []

        # ----------------------- branch: Number‑of‑Papers vs. PDF‑ZIP upload
        if self.input_type_selector.value == "Number of Papers":
            children.extend(
                [
                    self.number_papers_input,
                    widgets.HTML("<b>Upload a query file (.txt):</b>"),
                    self.paper_query_chooser,
                ]
            )
        elif self.input_type_selector.value == "PDF Zip File":
            children.append(self.pdf_zip_chooser)

        # ----------------------- common inputs (conditionally incl. API chooser)
        children.extend(
            [
                self.keywords_chooser,
                self.ontology_chooser,
                self.ai_selector,
            ]
        )

        if self._needs_api_key(self.ai_selector.value):
            children.append(self.api_keys_chooser)

        children.extend(
            [
                self.kg_selector,
                self.credentials_chooser,
                self.output_dir_chooser,
                self.submit_button,
            ]
        )

        self.conditional_inputs.children = children

    # ---------------------------------------------------------------------
    # Submission handler ---------------------------------------------------
    # ---------------------------------------------------------------------
    def _on_submit_clicked(self, _button):
        with self.output:
            clear_output()

        errors = []

        # ---------------------------- stash simple params
        self.params.input_type = self.input_type_selector.value
        self.params.AI = self.ai_selector.value
        self.params.kg_type = self.kg_selector.value

        # ---------------------------- branch‑specific validation
        if self.params.input_type == "Number of Papers":
            if self.number_papers_input.value <= 0:
                errors.append("❌ Number of papers must be > 0.")
            else:
                self.params.number_papers = self.number_papers_input.value

            if not self.paper_query_chooser.selected:
                errors.append("❌ Please select a query file.")
            else:
                self.params.paper_query_file = self.paper_query_chooser.selected

        elif self.params.input_type == "PDF Zip File":
            if not self.pdf_zip_chooser.selected:
                errors.append("❌ Please select a ZIP file.")
            else:
                self.params.pdf_zip = self.pdf_zip_chooser.selected
        else:
            errors.append("❌ Please select a paper input type.")

        # ---------------------------- common validation
        if not self.keywords_chooser.selected:
            errors.append("❌ Keywords file not selected.")
        else:
            self.params.keywords_file = self.keywords_chooser.selected

        if not self.ontology_chooser.selected:
            errors.append("❌ Ontology file not selected.")
        else:
            self.params.ontology_file = self.ontology_chooser.selected

        # API‑keys only if needed ------------------------------------------
        if self._needs_api_key(self.params.AI):
            if not self.api_keys_chooser.selected:
                errors.append("❌ API keys file not selected.")
            else:
                self.params.api_keys_file = self.api_keys_chooser.selected
        else:
            self.params.api_keys_file = ""  # Explicitly blank for Ollama

        if not self.credentials_chooser.selected:
            errors.append("❌ KG credentials file not selected.")
        else:
            self.params.kg_credentials_file = self.credentials_chooser.selected

        if not self.output_dir_chooser.selected_path:
            errors.append("❌ Output directory not selected.")
        else:
            self.params.output_dir = self.output_dir_chooser.selected_path

        # ---------------------------- Report or display parameters --------
        with self.output:
            if errors:
                print("⚠️ Please fix the following issues:\n")
                for err in errors:
                    print(err)
                return

            print("✅ All inputs collected:")
            print("➡️ Input Type:", self.params.input_type)
            if self.params.input_type == "Number of Papers":
                print("  - Number of papers:", self.params.number_papers)
                print("  - Query file:", self.params.paper_query_file)
            else:
                print("  - PDF ZIP:", self.params.pdf_zip)

            print("📝 Keywords:", self.params.keywords_file)
            print("📖 Ontology:", self.params.ontology_file)
            if self._needs_api_key(self.params.AI):
                print("🔐 API Keys:", self.params.api_keys_file)
            else:
                print("🔐 API Keys: (not required for Ollama)")
            print("🧠 KG Credentials:", self.params.kg_credentials_file)
            print("🧪 AI:", self.params.AI)
            print("🧩 KG Type:", self.params.kg_type)
            print("📁 Output Dir:", self.params.output_dir)

    # ---------------------------------------------------------------------
    # Public method --------------------------------------------------------
    # ---------------------------------------------------------------------
    def display_ui(self):
        """Render all widgets in a VBox so the caller just *displays* this UI."""
        display(
            widgets.VBox(
                [self.input_type_selector, self.conditional_inputs, self.output]
            )
        )

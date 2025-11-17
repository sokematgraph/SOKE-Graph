import os
from types import SimpleNamespace
import ipywidgets as widgets
from IPython.display import clear_output, display
from ipyfilechooser import FileChooser

class SOKEGraphUI:
    """Interactive widget UI for configuring a SOKEGraph run inside Colab."""

    @staticmethod
    def _needs_api_key(ai_name: str) -> bool:
        return ai_name.lower() in {"openai", "gemini", "llama", "claude"}
    
    @staticmethod
    def _needs_cred(kg: str) -> bool:
        return kg.lower() in {"neo4j"}

    def __init__(self):
        self.params = SimpleNamespace(
            paper_source="",
            AI="",
            kg_type="",
            number_papers=None,
            paper_query_file="",
            pdf_zip="",
            journal_api_key_file="",
            keywords_file="",
            ontology_file="",
            api_keys_file="",
            kg_credentials_file="",
            output_dir="",
            field_of_interest="", 
        )

        # ──────────────────────── Primary Input: Paper Source ───────────────────────
        self.paper_source_selector = widgets.Dropdown(
            options=["Select Source", "Semantic Scholar", "PDF Zip File", "Journal API"],
            description="Paper Source:"
        )

        # ──────────────────────── Common widgets ───────────────────────
        self.number_papers_input = widgets.IntText(description="Num Papers/Entries:")

        def _make_fc(title, pattern, dirs=False):
            fc = FileChooser()
            fc.title = title
            fc.filter_pattern = pattern
            fc.show_only_dirs = dirs
            fc.use_dir_icons = dirs
            return fc

        self.query_file_chooser = _make_fc("📄 Select Query File (.txt)", "*.txt")
        self.pdf_zip_chooser = _make_fc("🗂 Select ZIP File of PDFs", "*.zip")
        self.journal_api_key_chooser = _make_fc("🔑 Select Journal API Key (.txt)", "*.txt")

        self.keywords_chooser = _make_fc("📝 Select Keywords File", "*.txt")
        self.ontology_chooser = _make_fc("📖 Select Ontology File", "*.json")
        self.field_of_interest_input = widgets.Text(
        description="Field of Interest:",
        placeholder="e.g., Bioinformatics, NLP, Healthcare...",
        value="Materials Science"   # 👈 default value
        )
        self.api_keys_chooser = _make_fc("🔐 Select AI API Keys File", "*.txt")
        self.credentials_chooser = _make_fc("🔐 Select KG Credentials File", "*.json")
        self.output_dir_chooser = _make_fc("📁 Select Output Directory", "*", dirs=True)

        self.ai_selector = widgets.Dropdown(
            options=["openAI", "gemini", "llama", "ollama", "claude"],
            description="AI Agent:"
        )

        self.kg_selector = widgets.Dropdown(
            options=["networkx", "neo4j"],
            description="KG Type:"
        )

        self.submit_button = widgets.Button(description="Submit ✅")
        self.output = widgets.Output()
        self.conditional_inputs = widgets.VBox([])

        # ──────────────────────── Event Bindings ───────────────────────
        self.paper_source_selector.observe(self._update_inputs, names="value")
        self.ai_selector.observe(self._update_inputs, names="value")
        self.kg_selector.observe(self._update_inputs, names="value")
        self.submit_button.on_click(self._on_submit_clicked)

    def _update_inputs(self, change=None):
        with self.output:
            clear_output()

        children = []

        paper_source = self.paper_source_selector.value

        if paper_source == "Semantic Scholar":
            children.extend([self.number_papers_input, self.query_file_chooser])
        elif paper_source == "PDF Zip File":
            children.append(self.pdf_zip_chooser)
        elif paper_source == "Journal API":
            children.extend([
                self.number_papers_input,
                self.query_file_chooser,
                self.journal_api_key_chooser
            ])

        children.extend([
            self.keywords_chooser,
            self.ontology_chooser,
            self.field_of_interest_input,
            self.ai_selector
        ])

        if self._needs_api_key(self.ai_selector.value):
            print("yesyes")
            children.append(self.api_keys_chooser)

        children.extend([self.kg_selector])
        
        if self._needs_cred(self.kg_selector.value):
            print("yes")
            children.append(self.credentials_chooser)
        else:
            print("no")
        children.extend([
            self.output_dir_chooser,
            self.submit_button
        ])

        self.conditional_inputs.children = children

    def _on_submit_clicked(self, _button):
        with self.output:
            clear_output()

        errors = []
        self.params.paper_source = self.paper_source_selector.value
        self.params.AI = self.ai_selector.value
        self.params.kg_type = self.kg_selector.value

        # ─────────────── Paper Source logic ───────────────
        if self.params.paper_source == "Semantic Scholar":
            if self.number_papers_input.value <= 0:
                errors.append("❌ Number of papers must be > 0.")
            else:
                self.params.number_papers = self.number_papers_input.value

            if not self.query_file_chooser.selected:
                errors.append("❌ Please select a query file.")
            else:
                self.params.paper_query_file = self.query_file_chooser.selected

        elif self.params.paper_source == "PDF Zip File":
            if not self.pdf_zip_chooser.selected:
                errors.append("❌ Please select a PDF ZIP file.")
            else:
                self.params.pdf_zip = self.pdf_zip_chooser.selected

        elif self.params.paper_source == "Journal API":
            if self.number_papers_input.value <= 0:
                errors.append("❌ Number of journals must be > 0.")
            else:
                self.params.number_papers = self.number_papers_input.value

            if not self.query_file_chooser.selected:
                errors.append("❌ Please select a query file.")
            else:
                self.params.paper_query_file = self.query_file_chooser.selected

            if not self.journal_api_key_chooser.selected:
                errors.append("❌ Journal API key file is required.")
            else:
                self.params.journal_api_key_file = self.journal_api_key_chooser.selected

        else:
            errors.append("❌ Please select a paper source.")

        # ─────────────── Common inputs ───────────────
        if not self.keywords_chooser.selected:
            errors.append("❌ Keywords file not selected.")
        else:
            self.params.keywords_file = self.keywords_chooser.selected

        if not self.ontology_chooser.selected:
            errors.append("❌ Ontology file not selected.")
        else:
            self.params.ontology_file = self.ontology_chooser.selected

        self.params.field_of_interest = self.field_of_interest_input.value

        if self._needs_api_key(self.params.AI):
            if not self.api_keys_chooser.selected:
                errors.append("❌ AI API keys file not selected.")
            else:
                self.params.api_keys_file = self.api_keys_chooser.selected

        if self._needs_cred(self.params.kg_type):
            if not self.credentials_chooser.selected:
                errors.append("❌ KG credentials file not selected.")
            else:
                self.params.kg_credentials_file = self.credentials_chooser.selected

        if not self.output_dir_chooser.selected_path:
            errors.append("❌ Output directory not selected.")
        else:
            self.params.output_dir = self.output_dir_chooser.selected_path

        with self.output:
            if errors:
                print("⚠️ Please fix the following issues:\n")
                for err in errors:
                    print(err)
                return

            print("✅ All inputs collected!")
            print("📄 Input Type:", self.params.paper_source)
            if self.params.paper_source in {"Semantic Scholar", "Journal API"}:
                print("  - Num Papers:", self.params.number_papers)
                print("  - Query File:", self.params.paper_query_file)
            if self.params.paper_source == "PDF Zip File":
                print("  - ZIP:", self.params.pdf_zip)
            if self.params.paper_source == "Journal API":
                print("  - Journal API Key:", self.params.journal_api_key_file)
            print("📖 Ontology:", self.params.ontology_file)
            print("🎯 Field of Interest:", self.params.field_of_interest or "(none)")
            print("📝 Keywords:", self.params.keywords_file)
            print("🧠 AI:", self.params.AI)
            print("🔐 AI Key File:", self.params.api_keys_file or "(not needed)")
            print("🧩 KG Type:", self.params.kg_type)
            print("🔐 KG Credentials:", self.params.kg_credentials_file)
            print("📁 Output Dir:", self.params.output_dir)

    def display_ui(self):
        display(widgets.VBox([
            self.paper_source_selector,
            self.conditional_inputs,
            self.output
        ]))

import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
from types import SimpleNamespace
import os

class SOKEGraphUI:
    """
    Google Colab-friendly UI for selecting paper input type, configuration files, and triggering SOKEGraph pipeline.
    """

    def __init__(self):
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
            output_dir=""
        )

        # Widgets
        self.input_type_selector = widgets.Dropdown(
            options=['Select Option', 'Number of Papers', 'PDF Zip File'],
            description='Paper Input Type:'
        )

        self.ai_selector = widgets.Dropdown(
            options=['openAI', 'gemini', 'llama', 'ollama'],
            description='AI Agent:'
        )

        self.kg_selector = widgets.Dropdown(
            options=['neo4j'],
            description='KG Type:'
        )

        self.number_papers_input = widgets.IntText(
            description='Number of Papers:'
        )

        self.paper_query_chooser = FileChooser()
        self.paper_query_chooser.title = "📄 Select Query File (.txt)"
        self.paper_query_chooser.filter_pattern = '*.txt'

        self.pdf_zip_chooser = FileChooser()
        self.pdf_zip_chooser.title = "🗂 Select ZIP File of PDFs"
        self.pdf_zip_chooser.filter_pattern = '*.zip'

        self.keywords_chooser = FileChooser()
        self.keywords_chooser.title = "📝 Select Keywords File"
        self.keywords_chooser.filter_pattern = '*.txt'

        self.ontology_chooser = FileChooser()
        self.ontology_chooser.title = "📖 Select Ontology File"
        self.ontology_chooser.filter_pattern = '*.json'

        self.api_keys_chooser = FileChooser()
        self.api_keys_chooser.title = "🔐 Select API Keys File"
        self.api_keys_chooser.filter_pattern = '*.txt'

        self.credentials_chooser = FileChooser()
        self.credentials_chooser.title = "🔐 Select KG Credentials File"
        self.credentials_chooser.filter_pattern = '*.json'

        self.output_dir_chooser = FileChooser()
        self.output_dir_chooser.title = "📁 Select Output Directory"
        self.output_dir_chooser.show_only_dirs = True
        self.output_dir_chooser.use_dir_icons = True

        self.submit_button = widgets.Button(description="Submit ✅")
        self.output = widgets.Output()
        self.conditional_inputs = widgets.VBox([])

        # Bind events
        self.input_type_selector.observe(self.update_inputs, names='value')
        self.submit_button.on_click(self.on_submit_clicked)

    def update_inputs(self, change=None):
        with self.output:
            clear_output()
            children = []

            if self.input_type_selector.value == 'Number of Papers':
                children = [
                    self.number_papers_input,
                    widgets.HTML("<b>Upload a query file (.txt):</b>"),
                    self.paper_query_chooser
                ]
            elif self.input_type_selector.value == 'PDF Zip File':
                children = [self.pdf_zip_chooser]

            # Common fields
            children += [
                self.keywords_chooser,
                self.ontology_chooser,
                self.ai_selector,
                self.api_keys_chooser,
                self.kg_selector,
                self.credentials_chooser,
                self.output_dir_chooser,
                self.submit_button
            ]

            self.conditional_inputs.children = children

    def on_submit_clicked(self, b):
        with self.output:
            clear_output()
            errors = []

            self.params.input_type = self.input_type_selector.value
            self.params.AI = self.ai_selector.value
            self.params.kg_type = self.kg_selector.value

            if self.params.input_type == 'Number of Papers':
                if self.number_papers_input.value <= 0:
                    errors.append("❌ Number of papers must be > 0.")
                else:
                    self.params.number_papers = self.number_papers_input.value

                if not self.paper_query_chooser.selected:
                    errors.append("❌ Please select a query file.")
                else:
                    self.params.paper_query_file = self.paper_query_chooser.selected

            elif self.params.input_type == 'PDF Zip File':
                if not self.pdf_zip_chooser.selected:
                    errors.append("❌ Please select a ZIP file.")
                else:
                    self.params.pdf_zip = self.pdf_zip_chooser.selected
            else:
                errors.append("❌ Please select a paper input type.")

            # Common validations
            if not self.keywords_chooser.selected:
                errors.append("❌ Keywords file not selected.")
            else:
                self.params.keywords_file = self.keywords_chooser.selected

            if not self.ontology_chooser.selected:
                errors.append("❌ Ontology file not selected.")
            else:
                self.params.ontology_file = self.ontology_chooser.selected

            if not self.api_keys_chooser.selected:
                errors.append("❌ API keys file not selected.")
            else:
                self.params.api_keys_file = self.api_keys_chooser.selected

            if not self.credentials_chooser.selected:
                errors.append("❌ KG credentials file not selected.")
            else:
                self.params.kg_credentials_file = self.credentials_chooser.selected

            if not self.output_dir_chooser.selected_path:
                errors.append("❌ Output directory not selected.")
            else:
                self.params.output_dir = self.output_dir_chooser.selected_path

            if errors:
                print("⚠️ Please fix the following issues:\n")
                for err in errors:
                    print(err)
                return

            print("✅ All inputs collected:")
            print("➡️ Input Type:", self.params.input_type)
            if self.params.input_type == 'Number of Papers':
                print("  - Number of papers:", self.params.number_papers)
                print("  - Query file:", self.params.paper_query_file)
            else:
                print("  - PDF ZIP:", self.params.pdf_zip)

            print("📝 Keywords:", self.params.keywords_file)
            print("📖 Ontology:", self.params.ontology_file)
            print("🔐 API Keys:", self.params.api_keys_file)
            print("🧠 KG Credentials:", self.params.kg_credentials_file)
            print("🧪 AI:", self.params.AI)
            print("🧩 KG Type:", self.params.kg_type)
            print("📁 Output Dir:", self.params.output_dir)

    def display_ui(self):
        display(widgets.VBox([
            self.input_type_selector,
            self.conditional_inputs,
            self.output
        ]))
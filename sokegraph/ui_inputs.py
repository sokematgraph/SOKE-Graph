import ipywidgets as widgets
from IPython.display import display
import os
from types import SimpleNamespace
from ipyfilechooser import FileChooser

class SOKEGraphUI:
    """
    Interactive UI class for configuring and submitting paper retrieval
    and knowledge graph processing parameters using ipywidgets and ipyfilechooser.

    This class manages:
    - Selection of input type (number of papers or PDF zip)
    - Choosing related files and directories via file choosers
    - Validation and display of form submission status
    """

    def __init__(self):
        # Create a temporary directory for inputs (if needed)
        self.temp_dir = "tmp/sokegraph_inputs"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Parameters object to hold user selections/inputs
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

        # Dropdown widget for selecting paper input type
        self.input_type_selector = widgets.Dropdown(
            options=['Select Option', 'Number of Papers', 'PDF Zip File'],
            description='Paper Input Type:',
            style={'description_width': 'initial'}
        )

        # Dropdown widget for selecting AI agent to use
        self.ai_selector = widgets.Dropdown(
            options=['openAI', 'gemini', 'llama'],
            description='AI Agent:',
            style={'description_width': 'initial'}
        )

        # Dropdown widget for selecting Knowledge Graph type (currently only neo4j)
        self.kg_selector = widgets.Dropdown(
            options=['neo4j'],
            description='Knowledge Graph:',
            style={'description_width': 'initial'}
        )

        # Input for number of papers (shown conditionally)
        self.number_papers_input = widgets.IntText(
            description='Number of Papers:',
            style={'description_width': 'initial'}
        )

        # FileChooser widgets for selecting relevant files

        # Query file chooser: allows selection of text files containing paper queries
        self.paper_query_chooser = FileChooser()
        self.paper_query_chooser.title = 'Select Query File (.txt)'
        self.paper_query_chooser.filter_pattern = ['*.txt']

        # PDF zip file chooser for uploading ZIP archive of PDFs
        self.pdf_zip_chooser = FileChooser()
        self.pdf_zip_chooser.title = 'Select PDF ZIP File'
        self.pdf_zip_chooser.filter_pattern = ['*.zip']

        # Keywords file chooser (text files)
        self.keywords_chooser = FileChooser()
        self.keywords_chooser.title = 'Select Keywords File (.txt)'
        self.keywords_chooser.filter_pattern = ['*.txt']

        # Ontology JSON file chooser
        self.ontology_chooser = FileChooser()
        self.ontology_chooser.title = 'Select Ontology File (.json)'
        self.ontology_chooser.filter_pattern = ['*.json']

        # API keys file chooser (text files)
        self.api_keys_chooser = FileChooser()
        self.api_keys_chooser.title = 'Select API Keys File (.txt)'
        self.api_keys_chooser.filter_pattern = ['*.txt']

        # Knowledge graph credentials file chooser (JSON)
        self.credentials_chooser = FileChooser()
        self.credentials_chooser.title = 'Select KG Credentials File (.json)'
        self.credentials_chooser.filter_pattern = ['*.json']

        # Output directory chooser: configured to select directories only
        self.output_dir_chooser = FileChooser(use_dir_icons=True)
        self.output_dir_chooser.title = 'Select Output Directory'
        self.output_dir_chooser.show_only_dirs = True

        # Container to hold conditional inputs that change based on paper input type
        self.conditional_inputs = widgets.VBox([])

        # Output widget for displaying validation messages, errors, and status
        self.output = widgets.Output()

        # Submit button to validate inputs and submit the form
        self.submit_button = widgets.Button(description='Submit')

        # Observe changes to the input_type_selector dropdown and update UI accordingly
        self.input_type_selector.observe(self.update_inputs, names='value')

        # Bind submit button click event to handler method
        self.submit_button.on_click(self.on_submit_clicked)

    def update_inputs(self, change=None):
        """
        Updates visible input widgets dynamically based on selected input type.

        - Shows either number of papers + query file chooser, or
          PDF zip file chooser.
        - Always shows common file choosers and selectors.
        """
        with self.output:
            self.output.clear_output()  # Clear previous messages
            inputs = []

            # Show inputs conditionally based on selected paper input type
            if self.input_type_selector.value == 'Number of Papers':
                inputs = [self.number_papers_input, self.paper_query_chooser]
            elif self.input_type_selector.value == 'PDF Zip File':
                inputs = [self.pdf_zip_chooser]

            # Add common inputs that are always shown
            inputs += [
                self.keywords_chooser,
                self.ontology_chooser,
                self.ai_selector,
                self.api_keys_chooser,
                self.kg_selector,
                self.credentials_chooser,
                self.output_dir_chooser
            ]

            # Update the VBox container children to reflect inputs
            self.conditional_inputs.children = inputs

    def on_submit_clicked(self, b):
        """
        Handler for submit button click.

        - Validates all user inputs.
        - If any errors, displays messages and aborts.
        - If all valid, prints summary of chosen options.
        """
        with self.output:
            self.output.clear_output()  # Clear previous output messages
            errors = []

            # Capture form selections into params namespace
            self.params.input_type = self.input_type_selector.value
            self.params.AI = self.ai_selector.value
            self.params.kg_type = self.kg_selector.value

            # Validate inputs conditionally based on paper input type
            if self.params.input_type == 'Number of Papers':
                if self.number_papers_input.value is None or self.number_papers_input.value <= 0:
                    errors.append("❌ Number of papers must be greater than 0.")
                self.params.number_papers = self.number_papers_input.value

                if not self.paper_query_chooser.selected:
                    errors.append("❌ Please select a query file (.txt).")
                else:
                    self.params.paper_query_file = self.paper_query_chooser.selected

            elif self.params.input_type == 'PDF Zip File':
                if not self.pdf_zip_chooser.selected:
                    errors.append("❌ Please select a ZIP file containing PDFs.")
                else:
                    self.params.pdf_zip = self.pdf_zip_chooser.selected
            else:
                errors.append("❌ Please select a paper input type.")

            # Validate required common files
            if not self.keywords_chooser.selected:
                errors.append("❌ Please select a keywords file (.txt).")
            else:
                self.params.keywords_file = self.keywords_chooser.selected

            if not self.ontology_chooser.selected:
                errors.append("❌ Please select an ontology file (.json).")
            else:
                self.params.ontology_file = self.ontology_chooser.selected

            if not self.api_keys_chooser.selected:
                errors.append("❌ Please select an API keys file (.txt).")
            else:
                self.params.api_keys_file = self.api_keys_chooser.selected

            if not self.credentials_chooser.selected:
                errors.append("❌ Please select KG credentials file (.json).")
            else:
                self.params.kg_credentials_file = self.credentials_chooser.selected

            if not self.output_dir_chooser.selected_path:
                errors.append("❌ Please select a directory for output.")
            else:
                self.params.output_dir = self.output_dir_chooser.selected_path

            # If there are validation errors, print all and abort submission
            if errors:
                print("⚠️ Please fix the following issues before submitting:\n")
                for err in errors:
                    print(err)
                return

            # If all validations pass, print a success message and the selections summary
            print("✅ All inputs validated. Form Submitted Successfully:\n")
            print("➡️ Input Type:", self.params.input_type)
            if self.params.input_type == 'Number of Papers':
                print("  - Number of papers:", self.params.number_papers)
                print("  - Query File:", self.params.paper_query_file)
            elif self.params.input_type == 'PDF Zip File':
                print("  - ZIP File:", self.params.pdf_zip)

            print("📄 Keywords:", self.params.keywords_file)
            print("📖 Ontology:", self.params.ontology_file)
            print("🔐 API Keys:", self.params.api_keys_file)
            print("🗂 KG Credentials:", self.params.kg_credentials_file)
            print("🤖 AI Agent:", self.params.AI)
            print("🧠 Knowledge Graph:", self.params.kg_type)
            print("📁 Output Directory:", self.params.output_dir)

    def display_ui(self):
        """
        Displays the UI widgets in a vertical layout.

        Should be called to show the full interactive UI.
        """
        display(widgets.VBox([
            self.input_type_selector,
            self.conditional_inputs,
            self.submit_button,
            self.output
        ]))
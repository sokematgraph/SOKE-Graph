
# 🛠 INSTALLATION.md

This file provides detailed instructions for installing required tools to run **SOKEGraph**.

---

## 📑 Table of Contents

- [Install Git](#1-install-git)
- [Install Visual Studio Code (VS Code)](#2-install-visual-studio-code-vs-code)
  - [Add VS Code to PATH (Optional but Recommended)](#add-vs-code-to-path-optional-but-recommended)
  - [Install Extensions](#install-extensions)
- [Install Conda (Recommended for Virtual Environment)](#3-install-conda-recommended-for-virtual-environment)

---

## 1. Install Git
Git is required to clone the repository.

- **Windows:** [Download Git for Windows](https://git-scm.com/download/win) and follow the setup wizard.  
- **macOS:** Git is preinstalled. If not, install via [Homebrew](https://brew.sh/):  
  ```bash
  brew install git
  ```
- **Linux:** Use your package manager. For example (Ubuntu/Debian):  
  ```bash
  sudo apt update && sudo apt install git
  ```

---

## 2. Install Visual Studio Code (VS Code)

VS Code is the recommended editor for working with this project.

- **Download VS Code:** [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)
- Follow the installation wizard for your operating system.

### Add VS Code to PATH (Optional but Recommended)
During installation, check the box **"Add to PATH"** so you can open projects with `code .` from the terminal.

### Install Extensions
After installing VS Code, open it and install these extensions:
- **Python** (Microsoft)
- **Jupyter** (Microsoft)

These extensions make it easier to run and edit `.py` or `.ipynb` files directly inside VS Code.

---

## 3. Install Conda (Recommended for Virtual Environment)
We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight Conda).

- Download Miniconda for your OS.  
- Install and follow on-screen instructions.  

Verify installation:
```bash
conda --version
```

---

✅ With Git, VS Code, and Conda installed, you are ready to follow the steps in the [README](README.md) to set up and run the project.
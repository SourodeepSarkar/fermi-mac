# Fermi

**Fermi** is a lightweight, high-performance command-line study assistant built specifically for undergraduate physics workflows. Powered by the **Google Gemini 3.6 Flash API**, it provides real-time response streaming, session-based conversation persistence, directory/codebase snapshot caching, custom terminal LaTeX rendering, and native file attachments for research papers, diagrams, and problem sets.

---

## Key Features

* **Real-Time Streaming Output:** Generates instant live terminal responses with standard Markdown and custom LaTeX rendering for inline and block equations.
* **Non-Intrusive Terminal UI:** Features an animated thinking spinner while waiting for initial API chunks and completely suppresses SDK setup/warning logs.
* **On-Demand Response Copying:** Keep your clipboard clean during normal use, or use `/copy` to instantly capture the last generated response as raw Markdown.
* **Isolated Environment:** Entirely self-contained—dependencies, chat histories, command-line memory, and snapshot caches live within the project root.
* **Persistent Session Memory:** Organize discussions into named topics (e.g., `quantum_hw2`, `mechanics_lab`). Sessions persist across terminal restarts.
* **Codebase & Directory Indexing:** Supply entire code repositories (`.py`, `.cpp`, `.tex`, etc.) for immediate analysis with automated change-detection caching.
* **Multimodal File Support:** Upload PDFs, images, circuit diagrams, and data files via Gemini's File API.
* **System Global Command:** Access Fermi instantly from any directory using the `fermi` command.

---

## Project Structure

```text
fermi/
├── .env                  # Secret API key configuration (git-ignored)
├── .gitignore            # Git exclusion rules
├── .prompt_history       # Command-line input history (git-ignored)
├── cache/                # Codebase snapshot hashes (git-ignored)
├── history/              # Saved chat session JSONs (git-ignored)
├── main.py               # Core application logic & pipeline engine
├── requirements.txt      # Python dependencies
├── run.sh                # Executable wrapper & environment activator
└── venv/                 # Isolated Python virtual environment (git-ignored)

```

---

## Setup & Installation

### 1. Prerequisites

* Python 3.9 or higher
* A Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Environment Configuration

Clone or navigate into your local repository and set up the `.env` secret file:

```bash
cd fermi
echo 'GEMINI_API_KEY="your_actual_gemini_api_key_here"' > .env

```

### 3. Installation

Initialize the virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Make launcher executable
chmod +x run.sh

```

Ensure your `requirements.txt` contains:

```text
google-genai
python-dotenv
rich
prompt_toolkit
pyperclip

```

### 4. Global Alias Setup

To run `fermi` from any folder in your terminal, add an alias to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
echo 'alias fermi="'"$(pwd)/run.sh"'"' >> ~/.zshrc
source ~/.zshrc

```

---

## Usage Guide

### Starting Fermi

Run the global command from any directory:

```bash
fermi

```

Upon launching, you will be prompted to supply a session name:

```text
Session name (default: 'main_study'): quantum_mechanics

```

### Interactive CLI Commands

| Command | Usage | Description |
| --- | --- | --- |
| `/attach <file_path>` | `/attach ./lab_report.pdf` | Queues a file (PDF, image, data file) to send with your next prompt. |
| `/dir <dir_path>` | `/dir ./src/simulation` | Queues an entire directory to index into context with your next prompt. |
| `/copy` | `/copy` | Copies the last assistant response to the system clipboard as raw Markdown. |
| `exit` / `quit` | `exit` | Saves the session state to `history/` and closes the application. |

---

## Architecture & Development Details

* **Engine:** Built on the official `google-genai` SDK using `gemini-3.6-flash`.
* **Streaming & UI Mechanics:**
* Uses `rich.status.Status` for a non-blocking initial response spinner.
* Uses `rich.live.Live` and `rich.markdown.Markdown` for output rendering at 12 refreshes/second.
* Overrides `google_genai` logger severity to suppress AFC warnings during output execution.


* **Terminal Math Preprocessor:** Converts raw LaTeX constructs ($\nabla$, $\mathcal{P}$, $\mathbb{R}^3$, matrices, etc.) into clean Unicode symbols before standard Markdown parsing.
* **Caching Strategy:** The `read_directory()` function generates SHA-256 hashes based on absolute paths and modification times (`st_mtime`). Repeated queries on unmodified directories load instantly from `cache/`.
* **Prompt UI:** Implements `prompt_toolkit` to handle multiline inputs and persistent terminal command history.
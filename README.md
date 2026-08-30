# Fermi

**Fermi** is a lightweight, high-performance command-line study assistant built specifically for undergraduate physics workflows. Powered by the **Google Gemini 2.5 Flash API**, it provides real-time response streaming, session-based conversation persistence, directory/codebase snapshot caching, and native file attachments for research papers, diagrams, and problem sets.

---

## Key Features

* **Real-Time Streaming Output:** Generates instant live terminal responses with full Markdown and LaTeX math support (`$...$` or `$$...$$`).
* **Isolated Environment:** Entirely self-contained—all dependencies, chat histories, command-line memory, and snapshot caches live within the project root.
* **Persistent Session Memory:** Organize discussions into named topics (e.g., `quantum_hw2`, `mechanics_lab`). Sessions persist across terminal restarts.
* **Codebase & Directory Indexing:** Supply entire code repositories (`.py`, `.cpp`, `.tex`, etc.) for immediate analysis with automated change-detection caching.
* **Multimodal File Support:** Upload PDFs, images, circuit paths, and data files via Gemini's File API.
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
| `exit` / `quit` | `exit` | Saves the session state to `history/` and closes the application. |

### Example Workflow

```text
[You] > /attach problem_set_3.pdf
Queued attachment: problem_set_3.pdf

[You] > Solve Problem 2(b) step-by-step and write out the explicit Hamiltonian derivation.

```

```text
[You] > /dir ./stochastic_sim
Queued directory: ./stochastic_sim

[You] > Analyze my Python integration loop inside solver.py and suggest performance optimizations.

```

---

## Architecture & Development Details

* **Engine:** Built on the official `google-genai` SDK using `gemini-2.5-flash`.
* **Streaming & UI:** Leverages `rich.live.Live` and `rich.markdown.Markdown` for smooth output rendering at 12 refreshes/second.
* **Caching Strategy:** The `read_directory()` function generates SHA-256 hashes based on absolute paths and modification times (`st_mtime`). Repeated queries on unmodified directories load instantly from `cache/`.
* **Prompt UI:** Implements `prompt_toolkit` to handle multiline inputs and persistent terminal command history.

---

## Modifying & Extending

To customize system instructions or alter default parameters, edit `main.py`:

* **Change System Persona:** Update `sys_instruction` inside `PhysicsPipeline.stream_query()`.
* **Adjust Model Generation:** Modify `types.GenerateContentConfig(temperature=0.2)` to adjust creativity or precision.
* **File Extensions:** Extend the supported extensions list in `read_directory()` to include custom file types.

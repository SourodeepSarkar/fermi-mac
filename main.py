#!/usr/bin/env python3
"""
Fermi — an adaptive terminal study assistant for undergraduate physics,
powered by Google Gemini.

Design goals for this version:
  - Adaptive answers: short for quick questions, deep only when the problem
    actually calls for it — no reflexive scipy/numpy dumps.
  - Real tool use: Gemini's own web-search and code-execution tools do the
    numeric/lookup work instead of handing the user code to run themselves.
  - A CLI that feels like a real product: multiple sessions, model/temperature
    control, exportable transcripts, clean streamed rendering of text, code,
    and tool output.
"""
import os
import re
import sys
import json
import logging
import hashlib
import pathlib
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# Suppress Google GenAI AFC recommendation and log warnings
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ── Project-level isolated paths ─────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.resolve()
HISTORY_DIR = BASE_DIR / "history"
CACHE_DIR = BASE_DIR / "cache"
PROMPT_HIST_FILE = BASE_DIR / ".prompt_history"
CONFIG_FILE = BASE_DIR / "config.json"
load_dotenv(BASE_DIR / ".env")

HISTORY_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

console = Console()

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ALIASES = {
    "flash": "gemini-3.6-flash",   # fast, default
    "pro": "gemini-3.6-pro",       # heavier reasoning for hard problems
}
DEFAULT_CONFIG = {
    "model": MODEL_ALIASES["flash"],
    "temperature": 0.3,
    "web_search": False,
    "code_execution": False,
}


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


SYSTEM_INSTRUCTION = (
    "You are Fermi, a sharp, efficient physics study partner for undergraduates. "
    "Calibrate every answer to the question actually asked:\n"
    "- Quick factual or conceptual questions get a short, direct answer — a few "
    "sentences, not a lecture.\n"
    "- Only produce a full step-by-step derivation when the problem genuinely "
    "requires one, and stop once it's solved. Don't pad with extra sections, "
    "alternate methods, or 'further reading' unless the user asks for them.\n"
    "- Never dump boilerplate numerical code (scipy/numpy scaffolding, plotting "
    "setups, etc.) into your answer as a deliverable. If a number needs computing "
    "or a claim needs checking numerically, use the code execution tool yourself "
    "and report the result plainly — don't hand the user code to run unless they "
    "explicitly asked for an implementation.\n"
    "- Use the web search tool whenever a fact, constant, or reference needs to be "
    "current or precise, and briefly note where it came from.\n"
    "- Use LaTeX ($...$ or $$...$$) only for genuine mathematical expressions, "
    "never for plain text.\n"
    "- Default to plain, readable prose. Reach for tables or bullet lists only "
    "when they actually clarify something a paragraph wouldn't."
)


def format_terminal_latex(text: str) -> str:
    """Replaces raw LaTeX syntax with readable terminal-friendly math symbols."""
    replacements = {
        r'\mathcal{P}': '𝒫',
        r'\mathbb{R}^3': 'ℝ³',
        r'\mathbb{R}': 'ℝ',
        r'\mathbb{C}': 'ℂ',
        r'\in': '∈',
        r'\det': 'det',
        r'\times': '×',
        r'\cdot': '·',
        r'\vec': '⃗',
        r'\to': '→',
        r'\implies': '⇒',
        r'\iff': '⇔',
        r'\phi': 'ϕ',
        r'\Phi': 'Φ',
        r'\theta': 'θ',
        r'\Theta': 'Θ',
        r'\omega': 'ω',
        r'\Omega': 'Ω',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\Gamma': 'Γ',
        r'\delta': 'δ',
        r'\Delta': 'Δ',
        r'\epsilon': 'ε',
        r'\lambda': 'λ',
        r'\Lambda': 'Λ',
        r'\mu': 'μ',
        r'\sigma': 'σ',
        r'\Sigma': 'Σ',
        r'\pi': 'π',
        r'\hbar': 'ℏ',
        r'\partial': '∂',
        r'\nabla': '∇',
        r'\infty': '∞',
        r'\pm': '±',
        r'\leq': '≤',
        r'\geq': '≥',
        r'\approx': '≈',
        r'\neq': '≠',
        r'\langle': '⟨',
        r'\rangle': '⟩',
        r'\int': '∫',
        r'\sum': '∑',
    }
    for latex, unicode_sym in replacements.items():
        text = text.replace(latex, unicode_sym)

    text = re.sub(r'\\begin\{.*?\}(.*?)\\end\{.*?\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\pmatrix\{(.*?)\}', r'[\1]', text)
    return text


class PhysicsPipeline:
    def __init__(self, session_name: str = "main_study", config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else load_config()
        self.session_name = session_name
        self.session_file = HISTORY_DIR / f"{session_name}.json"
        self.last_response: str = ""
        self.last_prompt: str = ""
        self.extra_persona: str = ""

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable is not set.")
            console.print("[dim]Add it to a .env file next to main.py, e.g. GEMINI_API_KEY=your_key_here[/dim]")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self.history: List[Dict[str, str]] = self._load_history()

    @property
    def model_name(self) -> str:
        return self.config.get("model", MODEL_ALIASES["flash"])

    def _load_history(self) -> List[Dict[str, str]]:
        if self.session_file.exists():
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load history ({e}). Starting fresh.[/yellow]")
        return []

    def _save_history(self) -> None:
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _get_cache_path(self, content_key: str) -> pathlib.Path:
        file_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{file_hash}.txt"

    def attach_file(self, file_path: str) -> Optional[Any]:
        p = pathlib.Path(file_path).expanduser().resolve()
        if not p.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return None

        console.print(f"[cyan]Uploading attachment:[/cyan] {p.name}...")
        try:
            uploaded = self.client.files.upload(file=str(p))
            return uploaded
        except Exception as e:
            console.print(f"[bold red]Upload failed:[/bold red] {e}")
            return None

    def read_directory(self, dir_path: str, exts=(".py", ".cpp", ".h", ".tex", ".txt", ".md", ".csv")) -> str:
        p = pathlib.Path(dir_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            console.print(f"[bold red]Invalid directory:[/bold red] {dir_path}")
            return ""

        all_files = sorted([f for f in p.rglob("*") if f.suffix in exts and f.is_file()])
        mtimes = "".join([f"{f}:{f.stat().st_mtime}" for f in all_files])
        cache_file = self._get_cache_path(str(p) + mtimes)

        if cache_file.exists():
            console.print(f"[dim]Loaded directory snapshot from cache ({len(all_files)} files)[/dim]")
            return cache_file.read_text(encoding="utf-8")

        console.print(f"[yellow]Indexing directory ({len(all_files)} files)...[/yellow]")
        combined = []
        for file in all_files:
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
                rel_path = file.relative_to(p)
                combined.append(f"--- FILE: {rel_path} ---\n{content}\n--- END FILE ---")
            except Exception:
                continue

        result = "\n\n".join(combined)
        cache_file.write_text(result, encoding="utf-8")
        return result

    def _build_tools(self) -> List[types.Tool]:
        tools = []
        if self.config.get("web_search"):
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if self.config.get("code_execution"):
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
        return tools

    def _build_generation_config(self) -> types.GenerateContentConfig:
        instruction = SYSTEM_INSTRUCTION
        if self.extra_persona:
            instruction += f"\n\nStanding instruction from the user for this session:\n{self.extra_persona}"
        return types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=self.config.get("temperature", 0.3),
            tools=self._build_tools(),
        )

    def stream_query(self, prompt: str, attachment_path: Optional[str] = None,
                      dir_path: Optional[str] = None) -> bool:
        """Streams a response for `prompt`, rendering text, tool code, and tool
        output as they arrive. Returns True on success."""
        contents: List[Any] = []

        if dir_path:
            dir_context = self.read_directory(dir_path)
            if dir_context:
                contents.append(f"Directory context:\n{dir_context}\n\n")

        if attachment_path:
            file_obj = self.attach_file(attachment_path)
            if file_obj:
                contents.append(file_obj)

        full_context = []
        for msg in self.history:
            full_context.append(f"{msg['role'].capitalize()}: {msg['content']}")
        full_context.append(f"User: {prompt}")
        contents.extend(full_context)

        console.print()
        console.print(Rule(style="dim"))
        console.print("[bold magenta]Fermi[/bold magenta]")

        full_response = ""
        sources: List[str] = []

        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=self._build_generation_config(),
                )
                first_chunk = next(response_stream, None)

            def iter_chunks():
                if first_chunk is not None:
                    yield first_chunk
                for c in response_stream:
                    yield c

            live: Optional[Live] = None

            def ensure_live() -> Live:
                nonlocal live
                if live is None:
                    live = Live(console=console, refresh_per_second=12, transient=False)
                    live.start()
                return live

            def stop_live() -> None:
                nonlocal live
                if live is not None:
                    live.stop()
                    live = None

            for chunk in iter_chunks():
                if not chunk or not getattr(chunk, "candidates", None):
                    continue
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        text = getattr(part, "text", None)
                        code = getattr(part, "executable_code", None)
                        result = getattr(part, "code_execution_result", None)

                        if text:
                            full_response += text
                            ensure_live().update(Markdown(format_terminal_latex(full_response)))
                        elif code is not None:
                            stop_live()
                            code_str = getattr(code, "code", "") or ""
                            console.print(Panel(
                                Syntax(code_str.strip(), "python", theme="monokai", line_numbers=False),
                                title="[cyan]⚙ running a quick check[/cyan]",
                                border_style="cyan",
                            ))
                        elif result is not None:
                            stop_live()
                            output = (getattr(result, "output", "") or "").strip()
                            console.print(Panel(
                                output or "(no output)",
                                title="[green]✓ result[/green]",
                                border_style="green",
                            ))

                grounding = getattr(candidate, "grounding_metadata", None)
                chunks_meta = getattr(grounding, "grounding_chunks", None) if grounding else None
                if chunks_meta:
                    for gc in chunks_meta:
                        web = getattr(gc, "web", None)
                        if web and getattr(web, "uri", None):
                            label = getattr(web, "title", None) or web.uri
                            entry = f"{label} — {web.uri}" if getattr(web, "title", None) else web.uri
                            if entry not in sources:
                                sources.append(entry)

            stop_live()

            if sources:
                console.print()
                src_text = "\n".join(f"[dim]· {s}[/dim]" for s in sources[:5])
                console.print(Panel(src_text, title="[blue]sources[/blue]", border_style="blue"))

            self.last_response = full_response
            self.last_prompt = prompt
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": full_response})
            self._save_history()
            return True

        except Exception as e:
            console.print(f"\n[bold red]API Error:[/bold red] {e}")
            console.print("[dim]If this mentions incompatible tools, try /tools to disable one.[/dim]")
            return False


# ── CLI helpers ──────────────────────────────────────────────────────────

def list_sessions() -> List[str]:
    return sorted(p.stem for p in HISTORY_DIR.glob("*.json"))


def print_help() -> None:
    table = Table(title="Fermi Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Description")
    rows = [
        ("/attach <path>", "Queue a file (PDF, image, data) for your next message"),
        ("/dir <path>", "Queue a directory to index into context for your next message"),
        ("/clear", "Clear any queued attachment/directory"),
        ("/paste", "Enter multi-line input; finish with a line containing only /end"),
        ("/copy", "Copy the last response to your clipboard as raw Markdown"),
        ("/retry", "Resend your last message"),
        ("/history", "Show this session's conversation so far"),
        ("/export [path]", "Export this session to a Markdown file"),
        ("/sessions", "List all saved sessions"),
        ("/switch <name>", "Switch to (or create) another session"),
        ("/new <name>", "Start a brand-new session"),
        ("/model <flash|pro>", "Switch the Gemini model"),
        ("/temp <0.0-1.0>", "Set generation temperature"),
        ("/tools", "Show or toggle web search / code execution"),
        ("/persona <text>", "Add a standing instruction for this session"),
        ("/help", "Show this help"),
        ("/quit, /exit", "Save and exit"),
    ]
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)


def print_history(pipeline: PhysicsPipeline) -> None:
    if not pipeline.history:
        console.print("[dim]No messages yet in this session.[/dim]")
        return
    for msg in pipeline.history:
        role = "[bold blue]You[/bold blue]" if msg["role"] == "user" else "[bold magenta]Fermi[/bold magenta]"
        console.print(f"\n{role}:")
        console.print(Markdown(format_terminal_latex(msg["content"])))


def export_session(pipeline: PhysicsPipeline, path: Optional[str] = None) -> pathlib.Path:
    out_path = pathlib.Path(path).expanduser().resolve() if path else BASE_DIR / f"{pipeline.session_name}.md"
    lines = [
        f"# Fermi session: {pipeline.session_name}",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for msg in pipeline.history:
        speaker = "You" if msg["role"] == "user" else "Fermi"
        lines.append(f"## {speaker}\n\n{msg['content']}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def print_banner(config: Dict[str, Any]) -> None:
    tools_on = []
    if config.get("web_search"):
        tools_on.append("web search")
    if config.get("code_execution"):
        tools_on.append("code execution")
    
    tools_str = " + ".join(tools_on) if tools_on else "none (disabled by default)"

    console.print(Panel.fit(
        "[bold blue]Fermi[/bold blue] [dim]— adaptive physics study assistant[/dim]\n"
        f"[dim]Model: {config['model']}  ·  Tools: {tools_str}[/dim]\n"
        "[dim cyan]Note: web search and code execution are off by default to conserve rate limits. Use /tools to toggle.[/dim cyan]"
    ))


def main() -> None:
    config = load_config()
    print_banner(config)

    session_prompt = PromptSession(history=FileHistory(str(PROMPT_HIST_FILE)))

    existing = list_sessions()
    if existing:
        console.print(f"[dim]Existing sessions: {', '.join(existing)}[/dim]")
    session_name = session_prompt.prompt("Session name (default: 'main_study'): ").strip() or "main_study"
    pipeline = PhysicsPipeline(session_name=session_name, config=config)

    console.print(f"[bold green]Active session:[/bold green] {session_name}")
    console.print("[dim]Type /help for commands. Ask anything — Fermi adapts to how much you need.[/dim]\n")

    queued_attach: Optional[str] = None
    queued_dir: Optional[str] = None

    while True:
        try:
            user_input = session_prompt.prompt("\n[You] > ").strip()
            if not user_input:
                continue

            lower = user_input.lower()

            if lower in ("/quit", "/exit", "quit", "exit"):
                save_config(pipeline.config)
                console.print("[yellow]Session saved. Goodbye.[/yellow]")
                break

            if lower == "/help":
                print_help()
                continue

            if lower == "/copy":
                if pipeline.last_response:
                    pyperclip.copy(pipeline.last_response)
                    console.print("[green]✓ Copied last response to clipboard.[/green]")
                else:
                    console.print("[yellow]No response available to copy yet.[/yellow]")
                continue

            if lower == "/clear":
                queued_attach, queued_dir = None, None
                console.print("[green]Cleared queued attachment/directory.[/green]")
                continue

            if lower == "/history":
                print_history(pipeline)
                continue

            if lower == "/sessions":
                sessions = list_sessions()
                console.print("[bold]Sessions:[/bold] " + (", ".join(sessions) if sessions else "(none yet)"))
                continue

            if lower == "/retry":
                if not pipeline.last_prompt:
                    console.print("[yellow]Nothing to retry yet.[/yellow]")
                    continue
                pipeline.stream_query(pipeline.last_prompt, attachment_path=queued_attach, dir_path=queued_dir)
                queued_attach, queued_dir = None, None
                continue

            if lower == "/paste":
                console.print("[dim]Multi-line mode — finish with a line containing only /end[/dim]")
                lines = []
                while True:
                    line = session_prompt.prompt("... ")
                    if line.strip() == "/end":
                        break
                    lines.append(line)
                pasted = "\n".join(lines).strip()
                if not pasted:
                    continue
                pipeline.stream_query(pasted, attachment_path=queued_attach, dir_path=queued_dir)
                queued_attach, queued_dir = None, None
                continue

            if lower.startswith("/tools"):
                parts_ = user_input.split()
                if len(parts_) == 1:
                    console.print(
                        f"web search: {'on' if pipeline.config['web_search'] else 'off'}   "
                        f"code execution: {'on' if pipeline.config['code_execution'] else 'off'}"
                    )
                    console.print("[dim]Usage: /tools web on|off   or   /tools code on|off[/dim]")
                elif len(parts_) == 3 and parts_[1] in ("web", "code") and parts_[2] in ("on", "off"):
                    key = "web_search" if parts_[1] == "web" else "code_execution"
                    pipeline.config[key] = (parts_[2] == "on")
                    save_config(pipeline.config)
                    console.print(f"[green]{parts_[1]} tool {'enabled' if parts_[2] == 'on' else 'disabled'}.[/green]")
                else:
                    console.print("[yellow]Usage: /tools | /tools web on|off | /tools code on|off[/yellow]")
                continue

            if user_input.startswith("/attach "):
                queued_attach = user_input.split(" ", 1)[1].strip()
                console.print(f"[green]Queued attachment:[/green] {queued_attach}")
                continue

            if user_input.startswith("/dir "):
                queued_dir = user_input.split(" ", 1)[1].strip()
                console.print(f"[green]Queued directory:[/green] {queued_dir}")
                continue

            if lower.startswith("/export"):
                parts_ = user_input.split(maxsplit=1)
                path_arg = parts_[1].strip() if len(parts_) > 1 else None
                out = export_session(pipeline, path_arg)
                console.print(f"[green]Exported to:[/green] {out}")
                continue

            if user_input.startswith("/switch "):
                name = user_input.split(" ", 1)[1].strip()
                pipeline = PhysicsPipeline(session_name=name, config=pipeline.config)
                console.print(f"[bold green]Switched to session:[/bold green] {name}")
                continue

            if user_input.startswith("/new "):
                name = user_input.split(" ", 1)[1].strip()
                if (HISTORY_DIR / f"{name}.json").exists():
                    console.print(f"[yellow]Session '{name}' already exists — use /switch instead.[/yellow]")
                    continue
                pipeline = PhysicsPipeline(session_name=name, config=pipeline.config)
                console.print(f"[bold green]Started new session:[/bold green] {name}")
                continue

            if user_input.startswith("/model "):
                target = user_input.split(" ", 1)[1].strip().lower()
                if target in MODEL_ALIASES:
                    pipeline.config["model"] = MODEL_ALIASES[target]
                    save_config(pipeline.config)
                    console.print(f"[green]Model set to:[/green] {pipeline.config['model']}")
                else:
                    console.print(f"[yellow]Unknown model alias '{target}'. Options: {', '.join(MODEL_ALIASES)}[/yellow]")
                continue

            if user_input.startswith("/temp "):
                try:
                    val = max(0.0, min(1.0, float(user_input.split(" ", 1)[1].strip())))
                    pipeline.config["temperature"] = val
                    save_config(pipeline.config)
                    console.print(f"[green]Temperature set to:[/green] {val}")
                except ValueError:
                    console.print("[yellow]Usage: /temp 0.2[/yellow]")
                continue

            if user_input.startswith("/persona "):
                pipeline.extra_persona = user_input.split(" ", 1)[1].strip()
                console.print("[green]Added a standing instruction for this session.[/green]")
                continue

            if user_input.startswith("/"):
                console.print(f"[yellow]Unknown command:[/yellow] {user_input.split()[0]}  [dim](try /help)[/dim]")
                continue

            # Regular message
            pipeline.stream_query(
                prompt=user_input,
                attachment_path=queued_attach,
                dir_path=queued_dir,
            )
            queued_attach, queued_dir = None, None

        except (KeyboardInterrupt, EOFError):
            save_config(pipeline.config)
            console.print("\n[yellow]Session saved. Goodbye.[/yellow]")
            break


if __name__ == "__main__":
    main()
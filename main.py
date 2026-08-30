#!/usr/bin/env python3
import os
import re
import sys
import json
import logging
import hashlib
import pathlib
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import pyperclip
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# Fix 1: Suppress Google GenAI AFC recommendation and log warnings
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Project-level isolated paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
HISTORY_DIR = BASE_DIR / "history"
CACHE_DIR = BASE_DIR / "cache"
PROMPT_HIST_FILE = BASE_DIR / ".prompt_history"
load_dotenv(BASE_DIR / ".env")

HISTORY_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

console = Console()


def format_terminal_latex(text: str) -> str:
    """Fix 3: Replaces raw LaTeX syntax with readable terminal-friendly math symbols."""
    replacements = {
        r'\mathcal{P}': '𝒫',
        r'\mathbb{R}^3': 'ℝ³',
        r'\in': '∈',
        r'\det': 'det',
        r'\times': '×',
        r'\cdot': '·',
        r'\vec': '⃗',
        r'\to': '→',
        r'\implies': '⇒',
        r'\phi': 'ϕ',
        r'\Phi': 'Φ',
        r'\epsilon': 'ε',
        r'\partial': '∂',
        r'\nabla': '∇',
    }
    for latex, unicode_sym in replacements.items():
        text = text.replace(latex, unicode_sym)
    
    text = re.sub(r'\\begin\{.*?\}(.*?)\\end\{.*?\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\pmatrix\{(.*?)\}', r'[\1]', text)
    return text


class PhysicsPipeline:
    def __init__(self, session_name: str = "default"):
        self.session_name = session_name
        self.session_file = HISTORY_DIR / f"{session_name}.json"
        self.last_response: str = ""
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable is not set.")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"
        self.history: List[Dict[str, str]] = self._load_history()

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
        p = pathlib.Path(file_path).resolve()
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
        p = pathlib.Path(dir_path).resolve()
        if not p.exists() or not p.is_dir():
            console.print(f"[bold red]Invalid directory:[/bold red] {dir_path}")
            return ""

        all_files = sorted([f for f in p.rglob("*") if f.suffix in exts and f.is_file()])
        mtimes = "".join([f"{f}:{f.stat().st_mtime}" for f in all_files])
        cache_file = self._get_cache_path(str(p) + mtimes)

        if cache_file.exists():
            console.print(f"[dim]Loaded directory snapshot from cache ({len(all_files)} files)[/dim]")
            return cache_file.read_text(encoding="utf-8")

        console.print(f"[yellow]Indexing codebase directory ({len(all_files)} files)...[/yellow]")
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

    def stream_query(self, prompt: str, attachment_path: Optional[str] = None, dir_path: Optional[str] = None) -> None:
        contents = []

        if dir_path:
            dir_context = self.read_directory(dir_path)
            if dir_context:
                contents.append(f"Codebase Directory Context:\n{dir_context}\n\n")

        if attachment_path:
            file_obj = self.attach_file(attachment_path)
            if file_obj:
                contents.append(file_obj)

        sys_instruction = (
            "You are an elite physics assistant for undergraduate studies. "
            "Provide rigorous mathematical derivations using LaTeX notation ($...$ or $$...$$), "
            "highlight core physical assumptions, and give optimized code setups."
        )

        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=0.2
        )

        full_context = []
        for msg in self.history:
            full_context.append(f"{msg['role'].capitalize()}: {msg['content']}")
        full_context.append(f"User: {prompt}")

        contents.extend(full_context)

        full_response = ""
        console.print("\n[bold magenta]Fermi:[/bold magenta]")
        
        try:
            # Fix 2: Display spinner while initiating generation
            with console.status("[bold cyan]Fermi is thinking...", spinner="dots"):
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                first_chunk = next(response_stream, None)

            # Stream processing logic
            with Live(Markdown(""), console=console, refresh_per_second=12) as live:
                chunks = [first_chunk] if first_chunk else []
                for chunk in chunks:
                    if chunk and chunk.text:
                        full_response += chunk.text
                        live.update(Markdown(format_terminal_latex(full_response)))

                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        # Fix 3: Process text with LaTeX formatter before Markdown render
                        live.update(Markdown(format_terminal_latex(full_response)))

            self.last_response = full_response

            # Update history after successful stream
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": full_response})
            self._save_history()

        except Exception as e:
            console.print(f"\n[bold red]API Streaming Error:[/bold red] {e}")


def main():
    console.print(Panel.fit("[bold blue]Fermi AI Pipeline (Gemini 3.6 Flash Engine)[/bold blue]\n[dim]Isolated Environment & Fast Response Streaming[/dim]"))

    session_prompt = PromptSession(history=FileHistory(str(PROMPT_HIST_FILE)))
    
    session_name = session_prompt.prompt("Session name (default: 'main_study'): ").strip() or "main_study"
    pipeline = PhysicsPipeline(session_name=session_name)

    console.print(f"[bold green]Active Session:[/bold green] {session_name}")
    console.print("[dim]Commands: '/attach <path>' | '/dir <path>' | '/copy' | 'exit'[/dim]\n")

    queued_attach = None
    queued_dir = None

    while True:
        try:
            user_input = session_prompt.prompt("\n[You] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Saved session state. Exiting.[/yellow]")
                break

            # Manual copy via command only
            if user_input.lower() == "/copy":
                if pipeline.last_response:
                    pyperclip.copy(pipeline.last_response)
                    console.print("[green]✓ Copied last response to clipboard as raw Markdown.[/green]")
                else:
                    console.print("[yellow]No response available to copy yet.[/yellow]")
                continue

            if user_input.startswith("/attach "):
                queued_attach = user_input.split("/attach ", 1)[1].strip()
                console.print(f"[green]Queued attachment:[/green] {queued_attach}")
                continue

            if user_input.startswith("/dir "):
                queued_dir = user_input.split("/dir ", 1)[1].strip()
                console.print(f"[green]Queued directory:[/green] {queued_dir}")
                continue

            # Execute stream
            pipeline.stream_query(
                prompt=user_input,
                attachment_path=queued_attach,
                dir_path=queued_dir
            )

            # Reset single-use queues
            queued_attach = None
            queued_dir = None

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session saved.[/yellow]")
            break

if __name__ == "__main__":
    main()
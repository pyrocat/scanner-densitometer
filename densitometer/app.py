from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui.main_window import MainWindow


def _configure_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")


def main() -> None:
    root = tk.Tk()
    root.title("Scanner Densitometer")
    root.geometry("1380x860")
    root.minsize(1100, 720)
    _configure_style(root)

    window = MainWindow(root)
    window.grid(row=0, column=0, sticky="nsew")

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.mainloop()

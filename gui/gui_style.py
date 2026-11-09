"""Shared, interpreter-local Tk fonts and small GUI style defaults."""

from __future__ import annotations

import tkinter.font as tkfont
from tkinter import TclError


BODY_FONT = "AppBodyFont"
BODY_BOLD_FONT = "AppBodyBoldFont"
SMALL_FONT = "AppSmallFont"
SMALL_BOLD_FONT = "AppSmallBoldFont"
HEADING_FONT = "AppHeadingFont"
MONO_FONT = "AppMonoFont"

_FONT_SPECS = (
    (BODY_FONT, 10, "normal"),
    (BODY_BOLD_FONT, 10, "bold"),
    (SMALL_FONT, 9, "normal"),
    (SMALL_BOLD_FONT, 9, "bold"),
    (HEADING_FONT, 12, "bold"),
)
SANS_FAMILIES = (
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "PingFang TC",
)
_MONO_FAMILY_PREFERENCE = (
    "Consolas",
    "Cascadia Mono",
    "Menlo",
    "Monaco",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Noto Sans Mono",
)
_APP_FONT_NAMES = tuple(spec[0] for spec in _FONT_SPECS) + (MONO_FONT,)


def _family_lookup(families):
    return {str(family).casefold(): str(family) for family in families}


def choose_sans_family(available_names, fallback="sans-serif"):
    """Return the first installed preferred CJK sans family, or ``fallback``."""
    lookup = _family_lookup(available_names)
    for candidate in SANS_FAMILIES:
        family = lookup.get(candidate.casefold())
        if family:
            return family
    return fallback


def _choose_body_family(root, families=None):
    """Choose an installed Traditional Chinese family or Tk's real sans family."""
    if families is None:
        families = tkfont.families(root)
    # TkDefaultFont is a named font, so read its actual family instead of using
    # the name itself as a family in a font tuple.
    try:
        default_family = str(tkfont.nametofont("TkDefaultFont", root=root).actual("family"))
    except TclError:
        default_family = ""
    family = choose_sans_family(families, fallback=default_family or "")
    if family:
        return family
    if families:
        return str(families[0])
    return "sans-serif"


def _choose_mono_family(root, families=None):
    if families is None:
        families = tkfont.families(root)
    lookup = _family_lookup(families)
    try:
        fixed_family = str(tkfont.nametofont("TkFixedFont", root=root).actual("family"))
    except TclError:
        fixed_family = ""
    for candidate in _MONO_FAMILY_PREFERENCE:
        family = lookup.get(candidate.casefold())
        if family:
            return family
    if fixed_family:
        installed = lookup.get(fixed_family.casefold())
        if installed:
            return installed
    if fixed_family:
        return fixed_family
    return "monospace"


def _configure_named_font(root, name, family, size, weight="normal"):
    names = {str(font_name) for font_name in root.tk.call("font", "names")}
    options = ("-family", family, "-size", size, "-weight", weight)
    if name in names:
        root.tk.call("font", "configure", name, *options)
    else:
        root.tk.call("font", "create", name, *options)


def _cleanup_fonts_on_destroy(root):
    binding_id = None

    def cleanup(event):
        if event.widget is not root:
            return
        if binding_id:
            try:
                root.unbind("<Destroy>", binding_id)
            except TclError:
                pass
        try:
            existing = {str(name) for name in root.tk.call("font", "names")}
            for name in _APP_FONT_NAMES:
                if name in existing:
                    root.tk.call("font", "delete", name)
        except TclError:
            pass
        for attribute in ("_gui_style_applied", "_gui_style_binding_id"):
            if hasattr(root, attribute):
                delattr(root, attribute)

    binding_id = root.bind("<Destroy>", cleanup, add="+")
    root._gui_style_binding_id = binding_id


def configure_gui_style(root):
    """Install shared named fonts and ttk defaults before creating widgets.

    The font names are local to ``root``'s Tcl interpreter. Repeated calls on
    the same root are idempotent, and its destroy handler releases the names.
    Tk's scaling value is deliberately left untouched.
    """
    if getattr(root, "_gui_style_applied", False):
        return {
            "body": BODY_FONT,
            "small": SMALL_FONT,
            "small_bold": SMALL_BOLD_FONT,
            "heading": HEADING_FONT,
            "mono": MONO_FONT,
        }

    families = tkfont.families(root)
    body_family = _choose_body_family(root, families)
    mono_family = _choose_mono_family(root, families)
    for name, size, weight in _FONT_SPECS:
        _configure_named_font(root, name, body_family, size, weight)
    _configure_named_font(root, MONO_FONT, mono_family, 9)

    # Tk Canvas text and other native Tk controls use TkDefaultFont implicitly.
    # Giving it the selected real family keeps those defaults readable too.
    root.tk.call(
        "font", "configure", "TkDefaultFont",
        "-family", body_family, "-size", 10, "-weight", "normal",
    )
    root.option_add("*font", BODY_FONT, 80)
    # The Combobox list is a Tk Listbox popup and does not inherit ttk styles.
    root.option_add("*TCombobox*Listbox.font", BODY_FONT, 80)

    from tkinter import ttk

    style = ttk.Style(root)
    style.configure(".", font=BODY_FONT)
    style.configure("TLabel", foreground="#1f2937")
    style.configure("TButton", padding=(6, 2))
    style.configure("Accent.TButton", font=BODY_BOLD_FONT, foreground="#1d4ed8", padding=(8, 3))
    style.configure("Danger.TButton", foreground="#b42318", padding=(8, 3))
    for button_style in ("Accent.TButton", "Danger.TButton"):
        style.map(button_style, foreground=[("disabled", "#7b8491")])
    style.configure("Muted.TLabel", font=SMALL_FONT, foreground="#526174")
    style.configure("Status.TLabel", font=BODY_BOLD_FONT, foreground="#1f2937")
    style.configure("TLabelframe.Label", font=BODY_BOLD_FONT, foreground="#243554")
    style.configure("Section.TLabelframe", borderwidth=1, relief="groove")
    style.configure("Section.TLabelframe.Label", font=BODY_BOLD_FONT, foreground="#243554")
    style.configure("TNotebook.Tab", padding=(10, 5))
    style.configure("TCombobox", font=BODY_FONT, padding=(4, 2))
    line_space = int(root.tk.call("font", "metrics", BODY_FONT, "-linespace"))
    style.configure("Treeview", font=BODY_FONT, rowheight=line_space + 10,
                    background="#ffffff", fieldbackground="#ffffff", foreground="#1f2937")
    style.map("Treeview", background=[("selected", "#1d4ed8")],
              foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", font=SMALL_BOLD_FONT, padding=(7, 5))

    root._gui_style_applied = True
    _cleanup_fonts_on_destroy(root)
    return {
        "body": BODY_FONT,
        "small": SMALL_FONT,
        "small_bold": SMALL_BOLD_FONT,
        "heading": HEADING_FONT,
        "mono": MONO_FONT,
    }

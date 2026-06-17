"""
pyaitk.__main__  —  Pythonaibrain CLI dispatcher
=================================================
Invoked by the `pythonaibrain` and `pyaitk` console-script entry points,
and also by ``python -m pyaitk``.

Usage
-----
::

    pythonaibrain <group> <command> [options]
    pyaitk        <group> <command> [options]
    python -m pyaitk <group> <command> [options]

    pythonaibrain --version
    pythonaibrain --info
    pythonaibrain --modules

Groups & commands
-----------------
zentraa
    server      Start the ZENTRAA encrypted TCP chat server
    client      Connect as a chat client
    ai          Connect as the TIGER AI agent client
    web         Start the HTTP / WebSocket bridge

Examples
--------
::

    pythonaibrain zentraa server
    pythonaibrain zentraa server --host 0.0.0.0 --port 9999
    pythonaibrain zentraa server --config /path/to/ZENTRAA.pbcfg

    pythonaibrain zentraa client
    pythonaibrain zentraa client --host 127.0.0.1 --port 9999 --userid Alice
    pythonaibrain zentraa client --config /path/to/ZENTRAA.pbcfg

    pythonaibrain zentraa ai
    pythonaibrain zentraa ai --smart
    pythonaibrain zentraa ai --basic --host 127.0.0.1 --port 9999

    pythonaibrain zentraa web
    pythonaibrain zentraa web --http-port 7080 --tcp-host 127.0.0.1 --tcp-port 9999
    pythonaibrain zentraa web --no-auto-port --max-upload-mb 128 --history 1000

Author  : Divyanshu Sinha
Version : 1.1.9
License : LGPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable, NoReturn


# ---------------------------------------------------------------------------
# Version / metadata (mirrors pyaitk/__init__.py — kept local to avoid a
# heavy import just for --version)
# ---------------------------------------------------------------------------

_VERSION = "1.1.9"
_AUTHOR  = "Divyanshu Sinha"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _die(message: str, code: int = 2) -> NoReturn:
    print(f"pythonaibrain: error: {message}", file=sys.stderr)
    sys.exit(code)


def _lazy(module: str, attr: str) -> Callable[[], None]:
    """Return a zero-argument callable that imports *module* and calls *attr*().

    The import is deferred so that a missing optional dependency only errors
    when the relevant sub-command is actually invoked, not on every CLI call.
    """
    def _call() -> None:
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            _die(
                f"Required package for this command is not installed.\n"
                f"  {exc}\n\n"
                f"  Install it with:  pip install \"pythonaibrain[zentraa]\""
            )
        fn = getattr(mod, attr, None)
        if fn is None:
            _die(f"Entry point '{attr}' not found in module '{module}'.")
        fn()

    return _call


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------
# Structure:
#   _COMMANDS[group][command] = (module_path, function_name, description)

_COMMANDS: dict[str, dict[str, tuple[str, str, str]]] = {
    "zentraa": {
        "server": (
            "PyAgent.server",
            "main",
            "Start the ZENTRAA encrypted TCP chat server",
        ),
        "client": (
            "PyAgent.client",
            "main",
            "Connect to ZENTRAA as a chat client",
        ),
        "ai": (
            "PyAgent.clientAI",
            "main",
            "Connect as the TIGER AI agent (AdvanceBrain / Brain)",
        ),
        "web": (
            "PyAgent.http_server",
            "main",
            "Start the ZENTRAA HTTP / WebSocket bridge",
        ),
    },
}

# Per-command option hints shown in the help table
_CMD_OPTIONS: dict[str, dict[str, str]] = {
    "zentraa": {
        "server": "[--config F] [--host H] [--port P]",
        "client": "[--config F] [--host H] [--port P] [--userid U]",
        "ai":     "[--config F] [--host H] [--port P] [--smart | --basic]",
        "web":    "[--http-port P] [--tcp-host H] [--tcp-port P]\n"
                  "                      [--no-auto-port] [--max-upload-mb MB]\n"
                  "                      [--history N] [--ping-interval SECS]",
    },
}


# ---------------------------------------------------------------------------
# Top-level parser (handles --version / --info / --modules before dispatch)
# ---------------------------------------------------------------------------

def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythonaibrain",
        description=(
            "Pythonaibrain CLI — versatile AI toolkit\n"
            "Author: Divyanshu Sinha  |  v" + _VERSION
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=_build_help_epilog(),
    )
    parser.add_argument(
        "group",
        nargs="?",
        choices=list(_COMMANDS),
        metavar="<group>",
        help="Command group  (e.g. zentraa)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        metavar="<command>",
        help="Command within the group  (e.g. server)",
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Print package version and exit",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print package metadata and module availability, then exit",
    )
    parser.add_argument(
        "--modules",
        action="store_true",
        help="Print per-module availability table, then exit",
    )
    parser.add_argument(
        "--help", "-h",
        action="store_true",
        help="Show this help message and exit",
    )
    return parser


def _build_help_epilog() -> str:
    lines = [
        "",
        "Groups and commands:",
    ]
    for group, cmds in _COMMANDS.items():
        lines.append(f"  {group}")
        for cmd, (_mod, _fn, desc) in cmds.items():
            hint = _CMD_OPTIONS.get(group, {}).get(cmd, "")
            lines.append(f"    {cmd:<10}  {desc}")
            if hint:
                lines.append(f"              {hint}")
    lines += [
        "",
        "Examples:",
        "  pythonaibrain zentraa server",
        "  pythonaibrain zentraa server --host 0.0.0.0 --port 9999",
        "  pythonaibrain zentraa client --userid Alice",
        "  pythonaibrain zentraa ai --smart",
        "  pythonaibrain zentraa web --http-port 7080",
        "",
        "Each command passes all remaining flags directly to its entry point.",
        "Run  pythonaibrain <group> <command> --help  for command-level help.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --info / --modules helpers
# ---------------------------------------------------------------------------

def _print_version() -> None:
    print(f"pythonaibrain {_VERSION}")


def _print_info() -> None:
    try:
        import pyaitk  # noqa: PLC0415
        info = pyaitk.get_info()
    except Exception:
        info = {"version": _VERSION, "author": _AUTHOR, "modules": {}}

    print(f"Pythonaibrain {info['version']}")
    print(f"Author  : {info['author']}")
    _print_modules(info.get("modules", {}))


def _print_modules(modules: dict[str, bool] | None = None) -> None:
    if modules is None:
        try:
            import pyaitk  # noqa: PLC0415
            modules = pyaitk.check_module_availability()
        except Exception:
            modules = {}

    col_w = max((len(k) for k in modules), default=12) + 2
    print("\nModule availability:")
    print(f"  {'Module':<{col_w}}  Status")
    print(f"  {'-' * col_w}  ------")
    for name, available in sorted(modules.items()):
        status = "✔  available" if available else "✘  not installed"
        print(f"  {name:<{col_w}}  {status}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(group: str, command: str, rest: list[str]) -> None:
    """Look up (group, command), patch sys.argv, then call the entry point."""
    group_cmds = _COMMANDS.get(group)
    if group_cmds is None:
        _die(
            f"Unknown group '{group}'.\n"
            f"Available groups: {', '.join(_COMMANDS)}"
        )

    entry = group_cmds.get(command)
    if entry is None:
        available = ", ".join(group_cmds)
        _die(
            f"Unknown command '{command}' in group '{group}'.\n"
            f"Available commands: {available}"
        )

    module_path, fn_name, _ = entry

    # Rewrite sys.argv so the sub-command's own argparse sees the right prog
    # name and only the flags meant for it (not our group/command tokens).
    prog = f"pythonaibrain {group} {command}"
    sys.argv = [prog] + rest

    _lazy(module_path, fn_name)()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Primary entry point for the `pythonaibrain` / `pyaitk` CLI."""

    # ── Fast-path: no args → print help ─────────────────────────────────────
    if len(sys.argv) == 1:
        parser = _build_root_parser()
        parser.print_help()
        sys.exit(0)

    # ── Parse only the leading tokens; everything after <command> is "rest" ─
    # We use parse_known_args so that sub-command flags (e.g. --host) are
    # not consumed here and can be forwarded to the sub-command's own parser.
    parser = _build_root_parser()
    namespace, rest = parser.parse_known_args()

    # ── Global flags ─────────────────────────────────────────────────────────
    if namespace.help and not namespace.group:
        parser.print_help()
        sys.exit(0)

    if namespace.version:
        _print_version()
        sys.exit(0)

    if namespace.info:
        _print_info()
        sys.exit(0)

    if namespace.modules:
        _print_modules()
        sys.exit(0)

    # ── Group required from here ─────────────────────────────────────────────
    if not namespace.group:
        parser.print_help()
        sys.exit(0)

    group = namespace.group

    # ── Command required ─────────────────────────────────────────────────────
    if not namespace.command:
        group_cmds = _COMMANDS.get(group, {})
        print(f"pythonaibrain {group} — available commands:\n")
        for cmd, (_m, _f, desc) in group_cmds.items():
            hint = _CMD_OPTIONS.get(group, {}).get(cmd, "")
            print(f"  {cmd:<10}  {desc}")
            if hint:
                print(f"              {hint}")
        print(
            f"\nUsage:  pythonaibrain {group} <command> [options]\n"
            f"        pythonaibrain {group} <command> --help"
        )
        sys.exit(0)

    # If --help was given alongside a group+command, forward it to the
    # sub-command so its own argparse prints the right help text.
    if namespace.help:
        rest = ["--help"] + rest

    _dispatch(group, namespace.command, rest)


if __name__ == "__main__":
    main()
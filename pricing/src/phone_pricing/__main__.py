"""Run data preparation, model comparison, training and tuning commands."""

from __future__ import annotations

import argparse
import json
import runpy
import sys

from phone_pricing.common.project_paths import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    registry = json.loads(
        (PROJECT_ROOT.parent / "config/pricing_modules.json").read_text(encoding="utf-8")
    )
    parser = argparse.ArgumentParser(
        description=__doc__, usage="%(prog)s COMMAND [command arguments] | --list"
    )
    parser.add_argument("command", nargs="?", help="Pricing command name, without .py")
    parser.add_argument("--list", action="store_true", help="List available commands")
    if not arguments or arguments[0] in ("-h", "--help", "--list"):
        options = parser.parse_args(arguments)
        if options.list:
            print("\n".join(registry["commands"]))
        else:
            parser.print_help()
        return
    command = arguments[0].removesuffix(".py")
    if command not in registry["commands"]:
        parser.error(f"Unknown command: {arguments[0]}; use --list")
    module = registry["modules"][command + ".py"]
    sys.argv = [module, *arguments[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

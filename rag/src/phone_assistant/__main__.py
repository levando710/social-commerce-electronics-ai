"""Run research commands through their functional Python packages."""

from __future__ import annotations

import argparse
import runpy
import sys

from phone_assistant.common.project_layout import PYTHON_MODULES


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, usage="%(prog)s COMMAND [command arguments] | --list"
    )
    parser.add_argument("command", nargs="?", help="Research script name, without .py")
    parser.add_argument("--list", action="store_true", help="List modules by package")
    if not arguments or arguments[0] in ("-h", "--help", "--list"):
        options = parser.parse_args(arguments)
        if options.list:
            for module in sorted(PYTHON_MODULES.values()):
                print(module.removeprefix("phone_assistant."))
        else:
            parser.print_help()
        return
    name = arguments[0].removesuffix(".py") + ".py"
    module = PYTHON_MODULES.get(name)
    if module is None:
        parser.error(f"Unknown command: {arguments[0]}; use --list")
    sys.argv = [module, *arguments[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

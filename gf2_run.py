#!/usr/bin/env python
# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Compatibility wrapper for the installed :command:`qed-gf2` command."""

from gf2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

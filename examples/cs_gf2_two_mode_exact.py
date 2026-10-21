# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Two-mode exact CS-GF2; frequencies/couplings use atomic units."""

from _common import TWO_MODES, print_summary, run_example


if __name__ == "__main__":
    print_summary("Two-mode exact CS-GF2", run_example("std", TWO_MODES))

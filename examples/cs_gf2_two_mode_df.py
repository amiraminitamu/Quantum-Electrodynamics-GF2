# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Two-mode density-fitted CS-GF2; all inputs use atomic units."""

from _common import TWO_MODES, print_summary, run_example


if __name__ == "__main__":
    print_summary("Two-mode DF CS-GF2", run_example("df", TWO_MODES))

# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""One-mode exact CS-GF2; frequency/coupling use atomic units."""

from _common import ONE_MODE, print_summary, run_example


if __name__ == "__main__":
    print_summary("One-mode exact CS-GF2", run_example("std", ONE_MODE))

# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Electronic GF2 with exact AO ERIs; all energies are Hartree."""

from _common import print_summary, run_example


if __name__ == "__main__":
    print_summary("Electronic exact GF2", run_example("std", None))

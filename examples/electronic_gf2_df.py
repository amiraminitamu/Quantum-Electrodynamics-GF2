# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Electronic GF2 with cc-pVDZ-RI density fitting; energies are Hartree."""

from _common import print_summary, run_example


if __name__ == "__main__":
    print_summary("Electronic DF GF2", run_example("df", None))

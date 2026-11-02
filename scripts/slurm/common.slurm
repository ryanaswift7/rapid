set -euo pipefail

cd /project2/kpsounis_171/ryanswif/kd_first_principles

CONDA_BASE="/home1/ryanswif/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda init
conda activate rapid

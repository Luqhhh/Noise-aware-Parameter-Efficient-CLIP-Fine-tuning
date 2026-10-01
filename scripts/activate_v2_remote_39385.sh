# Source this file on the user-designated v2 CUDA server.
source /root/autodl-tmp/noise/venv/bin/activate
cd /root/autodl-tmp/noise/repo || return 1
export PYTHONPATH="$PWD/reproducibility/aegis_f1${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$PATH
export LC_ALL=C
export PYTHONNOUSERSITE=1;

if [ -f "${MAIN_ROOT}"/tools/activate_python.sh ]; then
    . "${MAIN_ROOT}"/tools/activate_python.sh
else
    echo "[INFO] "${MAIN_ROOT}"/tools/activate_python.sh is not present"
fi
. "${MAIN_ROOT}"/tools/extra_path.sh

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

if [ ! -v BACKEND ]; then
  BACKEND="espnet"
fi

if [ "$BACKEND" == "whisperx" ]; then
    eval "$(/share/homes/mengting7tw/miniconda3/bin/conda shell.bash hook)"
    conda activate whisperx
elif [ "$BACKEND" == "yourtts" ]; then
    eval "$(/share/homes/mengting7tw/miniconda3/bin/conda shell.bash hook)"
    conda activate yourtts
elif [ "$BACKEND" == "pyannote" ]; then
    eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
    conda activate pyannote
fi

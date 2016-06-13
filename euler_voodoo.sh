# Don't run this! Source it into your shell, or, better yet, have your nifty
# deploy script do that for you. You *are* using a deploy script, right?

module load eth_proxy gcc/4.9.2 python/3.3.3 openblas/0.2.13_par
export BLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export LAPACK=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export ATLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so

export C_INCLUDE_PATH="$C_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"

source ~/.venv/bin/activate

module load zlib

set -euxo pipefail
cd /outputs
gcc -O2 -fPIC -shared -pthread /outputs/ple_gather.c -o /outputs/libple_gather.so

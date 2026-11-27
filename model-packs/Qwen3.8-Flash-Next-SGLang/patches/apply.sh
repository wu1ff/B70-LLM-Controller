#!/usr/bin/env bash
# Apply the release to three clean pinned checkouts and stage the entrypoint.
set -euo pipefail
export LC_ALL=C
umask 022

if [[ $# != 4 ]]; then
    printf 'Usage: %s SGLANG_CHECKOUT XPU_CHECKOUT SYCL_TLA_CHECKOUT NEW_LAUNCH_DIR\n' "$0" >&2
    exit 2
fi

release=$(cd "$(dirname "$0")" && pwd)
campaign=$(dirname "$release")
sglang=$(realpath -e -- "$1")
xpu=$(realpath -e -- "$2")
tla=$(realpath -e -- "$3")
launch=$(realpath -m -- "$4")

require_base() {
    local tree=$1 revision=$2
    if [[ $(git -C "$tree" rev-parse --show-toplevel) != "$tree" ||
          $(git -C "$tree" rev-parse HEAD) != "$revision" ||
          -n $(git -C "$tree" status --porcelain --untracked-files=all) ]]; then
        printf 'Expected a clean checkout of %s at %s\n' "$revision" "$tree" >&2
        exit 1
    fi
}

apply() {
    local tree=$1 patch=$2
    git -C "$tree" apply --index --check --whitespace=nowarn "$patch"
    git -C "$tree" apply --index --whitespace=nowarn "$patch"
}

require_base "$sglang" ecd3d75a2a790cdb3acba0ecdb4786ff4d4ab25d
require_base "$xpu" 555588379c04d213388efaa043828560718a70f4
require_base "$tla" 7b56b936b87f799f80dde60c2fd8d11c1128cebb
test "$(git -C "$tla" write-tree)" = 2e19689c7f999d249c20aa1f08dbf0a8b80f8ae8
printf '%s  %s\n' c527bde1beb6470953bad73c5e265016854aa2246c38ddbc51604b59fc98feff \
    "$campaign/runtime/oci/entrypoint.sh" | sha256sum --check --strict
if [[ -e $launch || -L $launch ]]; then
    printf 'Expected a new launch directory at %s\n' "$launch" >&2
    exit 1
fi

for patch in "$campaign"/runtime/patches/????-intel-*.patch; do
    apply "$sglang" "$patch"
done
test "$(git -C "$sglang" write-tree)" = 1b7dbe8263fd2af9d319e418c231ab2291e06071

for patch in "$campaign"/runtime/patches/????-xpu-*.patch; do
    apply "$xpu" "$patch"
done
test "$(git -C "$xpu" write-tree)" = 8d0989dd7480899b43192fa36edebb8a3a820d9c

for patch in "$release"/000[1-4]-*.patch; do
    apply "$sglang" "$patch"
done
test "$(git -C "$sglang" write-tree)" = 1933fc7403a310166cc747b7e31243f17fd7642b

mkdir -- "$launch"
install -m 0755 "$campaign/runtime/oci/entrypoint.sh" "$launch/entrypoint.sh"
# Apply outside any surrounding repository to keep the staged entrypoint out of its index.
GIT_CEILING_DIRECTORIES=$(dirname "$launch") git -C "$launch" apply --check --whitespace=nowarn "$release/0005-oci-entrypoint.patch"
GIT_CEILING_DIRECTORIES=$(dirname "$launch") git -C "$launch" apply --whitespace=nowarn "$release/0005-oci-entrypoint.patch"
printf '%s  %s\n' e52b295908c51f806fabb21c23fac7de57995d72264d4603bd1530744471d7aa "$launch/entrypoint.sh" |
    sha256sum --check --strict

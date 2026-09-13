#!/bin/sh
# Release preparation for B70 LLM Controller.
#
#   scripts/release.sh           preparation mode: build and validate
#                                everything a release needs; the tag is only
#                                verified when it already exists
#   scripts/release.sh --strict  publish-ready gate: additionally requires
#                                tag v<VERSION> at HEAD
#
# Both modes require a clean tracked tree (untracked files are ignored) and
# never publish anything. Release artifacts land in dist/:
#   b70-llm-controller_<version>_amd64.deb, SHA256SUMS
set -eu
umask 022

. "$(dirname "$0")/librelease.sh"

usage() {
	echo "usage: scripts/release.sh [--strict]" >&2
	exit 2
}

strict=0
case ${1:-} in
"") ;;
--strict) strict=1 ;;
*) usage ;;
esac
[ $# -le 1 ] || usage

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

version=$(release_version)
release_valid_version "$version" || release_die "invalid VERSION '$version'"
debian_version=$(release_deb_version "$version")
tag=$(release_tag "$version")
deb=dist/$(release_deb_name "$version")
commit=$(git rev-parse HEAD)

echo "release: version $version (debian $debian_version)"
echo "release: commit $commit, expected tag $tag"

release_clean_tree "$root" ||
	release_die "tracked tree is dirty; commit or stash first (untracked files are ignored)"

tag_state=$(release_tag_state "$root" "$tag")
case $tag_state in
ok)
	echo "release: tag $tag verified at HEAD"
	;;
absent)
	[ "$strict" -eq 0 ] ||
		release_die "tag $tag not found; create it at HEAD for a publish-ready release"
	echo "release: tag $tag not present yet (preparation mode)"
	;;
mismatch)
	release_die "tag $tag does not point at HEAD ($commit)"
	;;
esac

echo "release: running clean build, tests, vet and packaging"
make clean build test vet package

built=$(./dist/b70ctl --version)
[ "$built" = "B70 LLM Controller $version" ] ||
	release_die "built dist/b70ctl --version: '$built'"

[ -f "$deb" ] || release_die "expected package $deb was not built"

check_field() {
	_got=$(dpkg-deb -f "$1" "$2")
	[ "$_got" = "$3" ] || release_die "package $2 field: expected '$3', got '$_got'"
}
check_field "$deb" Package b70-llm-controller
check_field "$deb" Version "$debian_version"
check_field "$deb" Architecture amd64

listing=$(dpkg-deb -c "$deb") || release_die "cannot list package contents"
echo "$listing" | grep -qE ' \./usr/bin/b70ctl$' ||
	release_die "package does not install /usr/bin/b70ctl"
echo "$listing" | grep -qE ' \./usr/share/doc/b70-llm-controller/LICENSE$' ||
	release_die "package does not include LICENSE"
echo "$listing" | grep -qE ' \./usr/share/doc/b70-llm-controller/README\.md$' ||
	release_die "package does not include README.md"

smoke=$(mktemp -d "${TMPDIR:-/tmp}/b70ctl-release.XXXXXX")
trap 'rm -rf "$smoke"' EXIT HUP INT TERM
dpkg-deb -x "$deb" "$smoke" || release_die "package does not extract cleanly"
packaged=$("$smoke/usr/bin/b70ctl" --version)
[ "$packaged" = "B70 LLM Controller $version" ] ||
	release_die "packaged b70ctl --version: '$packaged'"
built_sha=$(sha256sum dist/b70ctl | cut -d' ' -f1)
packaged_sha=$(sha256sum "$smoke/usr/bin/b70ctl" | cut -d' ' -f1)
[ "$built_sha" = "$packaged_sha" ] ||
	release_die "packaged binary differs from built dist/b70ctl"

release_write_sums dist "$(basename "$deb")"
(cd dist && sha256sum -c SHA256SUMS)

echo
echo "release: artifacts in dist/"
echo "  $(basename "$deb")"
echo "  SHA256SUMS"
sha256sum "$deb" | sed 's/^/  /'
echo "release: verify with: (cd dist && sha256sum -c SHA256SUMS)"
echo "release: nothing was published"

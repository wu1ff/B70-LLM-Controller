#!/bin/sh
set -eu
umask 022

. "$(dirname "$0")/librelease.sh"

project_version=$(release_version)
release_valid_version "$project_version" ||
	release_die "invalid VERSION '$project_version'"
debian_version=$(release_deb_version "$project_version")
package="b70-llm-controller_${debian_version}_amd64.deb"

# Reproducible bytes: clamp package timestamps to the source commit date
# (honored by dpkg-deb). Without it, archive mtimes track the build clock.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
	_epoch=$(git log -1 --format=%ct 2>/dev/null || true)
	if [ -n "$_epoch" ]; then
		SOURCE_DATE_EPOCH=$_epoch
		export SOURCE_DATE_EPOCH
	fi
fi

stage=$(mktemp -d "${TMPDIR:-/tmp}/b70ctl-package.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM

chmod 0755 "$stage"
install -d -m 0755 "$stage/DEBIAN"
install -d -m 0755 "$stage/usr/bin"
install -d -m 0755 "$stage/usr/share/doc/b70-llm-controller"
install -m 0755 dist/b70ctl "$stage/usr/bin/b70ctl"
install -m 0644 LICENSE "$stage/usr/share/doc/b70-llm-controller/LICENSE"
install -m 0644 README.md "$stage/usr/share/doc/b70-llm-controller/README.md"

cat > "$stage/DEBIAN/control" <<EOF
Package: b70-llm-controller
Version: $debian_version
Section: utils
Priority: optional
Architecture: amd64
Maintainer: wu1ff
Description: Simple TUI for installing and running qualified Intel Arc Pro B70 LLM model packs.
EOF

dpkg-deb --build --root-owner-group "$stage" "dist/$package"

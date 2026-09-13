#!/bin/sh
# Tests for the release helpers and release-preparation gates.
# Run via `make test-release`, which builds the package first.
set -eu
umask 022

. "$(dirname "$0")/librelease.sh"

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

passes=0
fails=0
ok() { echo " ok   - $1"; passes=$((passes + 1)); }
bad() { echo " FAIL - $1"; fails=$((fails + 1)); }

assert_eq() {
	_desc=$1 _got=$2 _want=$3
	if [ "$_got" = "$_want" ]; then
		ok "$_desc"
	else
		bad "$_desc (got '$_got', want '$_want')"
	fi
}

assert_succeeds() {
	_desc=$1
	shift
	if "$@" >/dev/null 2>&1; then
		ok "$_desc"
	else
		bad "$_desc (expected success)"
	fi
}

assert_fails() {
	_desc=$1
	shift
	if "$@" >/dev/null 2>&1; then
		bad "$_desc (expected failure)"
	else
		ok "$_desc"
	fi
}

echo "== version helpers =="

assert_succeeds "version 0.1.0-alpha.1 accepted" release_valid_version 0.1.0-alpha.1
assert_succeeds "version 0.1.0 accepted" release_valid_version 0.1.0
assert_succeeds "version 1.2.3-rc.4-2 accepted" release_valid_version 1.2.3-rc.4-2
assert_fails "empty version rejected" release_valid_version ""
assert_fails "version 0.1 rejected" release_valid_version 0.1
assert_fails "version 0.1.0.0 rejected" release_valid_version 0.1.0.0
assert_fails "v-prefixed version rejected" release_valid_version v0.1.0
assert_fails "trailing-dash version rejected" release_valid_version 0.1.0-
assert_fails "tilde version rejected (dash form required)" release_valid_version 0.1.0~alpha.1
assert_fails "underscore version rejected" release_valid_version 0.1.0_alpha.1
assert_fails "build-metadata version rejected" release_valid_version 0.1.0+build.1
assert_fails "space in version rejected" release_valid_version "0.1.0 alpha"

assert_eq "debian normalization 0.1.0-alpha.1" "$(release_deb_version 0.1.0-alpha.1)" "0.1.0~alpha.1"
assert_eq "debian normalization 0.1.0" "$(release_deb_version 0.1.0)" "0.1.0"
assert_eq "tag calculation" "$(release_tag 0.1.0-alpha.1)" "v0.1.0-alpha.1"
assert_eq "package filename calculation" "$(release_deb_name 0.1.0-alpha.1)" "b70-llm-controller_0.1.0~alpha.1_amd64.deb"

echo "== git gates =="

fixture=$(mktemp -d "${TMPDIR:-/tmp}/b70ctl-reltest.XXXXXX")
sumsdir=$(mktemp -d "${TMPDIR:-/tmp}/b70ctl-sumstest.XXXXXX")
trap 'rm -rf "$fixture" "$sumsdir"' EXIT HUP INT TERM
git -C "$fixture" init -q
git -C "$fixture" -c user.name=relay -c user.email=relay@example.invalid \
	commit -q --allow-empty -m one

assert_succeeds "clean tracked tree accepted" release_clean_tree "$fixture"
echo junk > "$fixture/untracked.txt"
assert_succeeds "untracked files still clean" release_clean_tree "$fixture"
echo junk > "$fixture/tracked.txt"
git -C "$fixture" add tracked.txt
assert_fails "staged change rejected" release_clean_tree "$fixture"
git -C "$fixture" -c user.name=relay -c user.email=relay@example.invalid \
	commit -q -m two
echo more >> "$fixture/tracked.txt"
assert_fails "unstaged modification rejected" release_clean_tree "$fixture"
git -C "$fixture" checkout -q -- tracked.txt

fixture_tag=$(release_tag 0.2.0)
assert_eq "missing tag reported absent" "$(release_tag_state "$fixture" "$fixture_tag")" "absent"
git -C "$fixture" tag "$fixture_tag"
assert_eq "tag at HEAD reported ok" "$(release_tag_state "$fixture" "$fixture_tag")" "ok"
git -C "$fixture" -c user.name=relay -c user.email=relay@example.invalid \
	commit -q --allow-empty -m three
assert_eq "tag on older commit reported mismatch" "$(release_tag_state "$fixture" "$fixture_tag")" "mismatch"

# Strict release check integration: with no matching tag in this repository
# it must refuse before building.
repo_tag=$(release_tag "$(release_version)")
if [ "$(release_tag_state "$root" "$repo_tag")" = absent ]; then
	if ./scripts/release.sh --strict >/dev/null 2>&1; then
		bad "strict release check rejects missing tag"
	else
		ok "strict release check rejects missing tag"
	fi
else
	ok "strict release check: $repo_tag present, missing-tag test skipped"
fi

echo "== package =="

version=$(release_version)
deb=dist/$(release_deb_name "$version")
if [ ! -f "$deb" ]; then
	echo "FAIL - package $deb missing (run: make package)" >&2
	exit 1
fi

assert_eq "package filename matches VERSION" "$(basename "$deb")" \
	"b70-llm-controller_$(release_deb_version "$version")_amd64.deb"
assert_eq "package name field" "$(dpkg-deb -f "$deb" Package)" "b70-llm-controller"
assert_eq "package version field" "$(dpkg-deb -f "$deb" Version)" "$(release_deb_version "$version")"
assert_eq "package architecture field" "$(dpkg-deb -f "$deb" Architecture)" "amd64"

listing=$(dpkg-deb -c "$deb")
if echo "$listing" | grep -qE ' \./usr/bin/b70ctl$'; then
	ok "package installs /usr/bin/b70ctl"
else
	bad "package installs /usr/bin/b70ctl"
fi
if echo "$listing" | grep -qE ' \./usr/share/doc/b70-llm-controller/LICENSE$'; then
	ok "package includes LICENSE"
else
	bad "package includes LICENSE"
fi
if echo "$listing" | grep -qE ' \./usr/share/doc/b70-llm-controller/README\.md$'; then
	ok "package includes README.md"
else
	bad "package includes README.md"
fi

smoke=$(mktemp -d "${TMPDIR:-/tmp}/b70ctl-pkgtest.XXXXXX")
dpkg-deb -x "$deb" "$smoke"
assert_eq "packaged b70ctl --version" "$("$smoke/usr/bin/b70ctl" --version)" \
	"B70 LLM Controller $version"
assert_eq "packaged binary matches built dist/b70ctl" \
	"$(sha256sum "$smoke/usr/bin/b70ctl" | cut -d' ' -f1)" \
	"$(sha256sum dist/b70ctl | cut -d' ' -f1)"
rm -rf "$smoke"

echo "== SHA256SUMS =="

cp "$deb" "$sumsdir/"
printf 'placeholder\n' > "$sumsdir/other-artifact"
release_write_sums "$sumsdir" "$(basename "$deb")" other-artifact
cp "$sumsdir/SHA256SUMS" "$sumsdir/first"
release_write_sums "$sumsdir" "$(basename "$deb")" other-artifact
if cmp -s "$sumsdir/first" "$sumsdir/SHA256SUMS"; then
	ok "SHA256SUMS regenerated deterministically"
else
	bad "SHA256SUMS regenerated deterministically"
fi
if head -n 1 "$sumsdir/SHA256SUMS" | grep -q " $(basename "$deb")$" &&
	head -n 2 "$sumsdir/SHA256SUMS" | tail -n 1 | grep -q ' other-artifact$'; then
	ok "SHA256SUMS entries in sorted order"
else
	bad "SHA256SUMS entries in sorted order"
fi
printf 'scratch\n' > "$sumsdir/scratch.txt"
release_write_sums "$sumsdir" "$(basename "$deb")" other-artifact
if grep -q scratch.txt "$sumsdir/SHA256SUMS"; then
	bad "scratch file excluded from SHA256SUMS"
else
	ok "scratch file excluded from SHA256SUMS"
fi
if (cd "$sumsdir" && sha256sum -c SHA256SUMS >/dev/null 2>&1); then
	ok "sha256sum -c passes"
else
	bad "sha256sum -c passes"
fi

release_write_sums dist "$(basename "$deb")"
if (cd dist && sha256sum -c SHA256SUMS >/dev/null 2>&1); then
	ok "dist SHA256SUMS verifies"
else
	bad "dist SHA256SUMS verifies"
fi

echo
if [ "$fails" -eq 0 ]; then
	echo "release tests: $passes passed, $fails failed"
	exit 0
fi
echo "release tests: $passes passed, $fails failed" >&2
exit 1

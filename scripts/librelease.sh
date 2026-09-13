# Shared release/packaging helpers for B70 LLM Controller.
# Sourced (never executed directly) by scripts/package_deb.sh,
# scripts/release.sh and scripts/release_test.sh. The VERSION file is the
# single version authority.

# Echo an error and exit nonzero from the sourcing script.
release_die() {
	echo "release: $*" >&2
	exit 1
}

# Echo the project version from the VERSION file (newlines stripped).
release_version() {
	[ -f VERSION ] || release_die "VERSION file not found"
	tr -d '\n' < VERSION
}

# Succeed only for X.Y.Z with an optional dot/hyphen separated prerelease
# (0.1.0, 0.1.0-alpha.1, 1.2.3-rc.4-2). Everything else is rejected.
release_valid_version() {
	printf '%s\n' "$1" | grep -Eq \
	    '^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9]+([.-][A-Za-z0-9]+)*)?$'
}

# Debian version normalization: 0.1.0-alpha.1 -> 0.1.0~alpha.1
release_deb_version() {
	printf '%s\n' "$1" | tr '-' '~'
}

# Release tag for a VERSION value: 0.1.0-alpha.1 -> v0.1.0-alpha.1
release_tag() {
	printf 'v%s\n' "$1"
}

# Expected package filename for a VERSION value.
release_deb_name() {
	printf 'b70-llm-controller_%s_amd64.deb\n' "$(release_deb_version "$1")"
}

# Succeed only when the tracked tree of git repo $1 is clean; untracked and
# ignored files do not matter.
release_clean_tree() {
	[ -z "$(git -C "$1" status --porcelain --untracked-files=no)" ]
}

# Echo the relationship of tag $2 to the HEAD of git repo $1:
# absent | ok | mismatch
release_tag_state() {
	_head=$(git -C "$1" rev-parse HEAD) || return 1
	_tagged=$(git -C "$1" rev-parse -q --verify "refs/tags/$2^{commit}" 2>/dev/null || true)
	if [ -z "$_tagged" ]; then
		echo absent
	elif [ "$_tagged" = "$_head" ]; then
		echo ok
	else
		echo mismatch
	fi
}

# Write $1/SHA256SUMS covering exactly the given filenames, sorted for
# deterministic order. Scratch files in $1 are never included because the
# distributable set is named explicitly by the caller.
release_write_sums() {
	_dist=$1
	shift
	printf '%s\n' "$@" | LC_ALL=C sort | while IFS= read -r _f; do
		(cd "$_dist" && sha256sum "$_f")
	done > "$_dist/SHA256SUMS.new"
	mv "$_dist/SHA256SUMS.new" "$_dist/SHA256SUMS"
}

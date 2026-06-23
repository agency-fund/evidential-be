#!/usr/bin/env bash
#
# Fail if a migration added on this branch sorts *before* a migration that already
# exists on the base branch.
#
# Atlas applies migration files in version (filename) order. Production has already
# applied every migration on the base branch, so a newly added file whose version is
# lower than the latest one on the base would be inserted *before* already-applied
# migrations -- a "non-linear" history that Atlas refuses to apply, breaking the deploy.
#
# This is a dependency-free stand-in for `atlas migrate lint`'s non_linear analyzer,
# which became Atlas Pro-only (requires `atlas login`) in Atlas v0.38. It needs only
# git, sort, and comm -- no dev database, no Docker, no Atlas login -- so it runs
# identically in CI and on a laptop.
#
# Usage: tools/check_migrations_linear.sh [base-ref]   (base-ref defaults to origin/main)
set -euo pipefail
export LC_ALL=C  # byte-wise sort/compare so it matches Atlas's filename ordering

BASE="${1:-origin/main}"
DIR="migrations/sa_postgres"

# Extract a migration's version: the filename without the .sql suffix or any _name part.
version_of() { sed -E 's/\.sql$//; s/_.*$//' <<<"$1"; }

# Migration basenames present on the base branch.
base_migrations() {
  git ls-tree -r --name-only "$BASE" -- "$DIR" 2>/dev/null \
    | grep -E '\.sql$' | xargs -n1 basename | sort
}

# Migration basenames present in the working tree (committed or not).
disk_migrations() {
  find "$DIR" -maxdepth 1 -name '*.sql' -exec basename {} \; | sort
}

# Highest version already on the base branch.
max_base="$(base_migrations | sed -E 's/\.sql$//; s/_.*$//' | sort | tail -1)"

status=0
while read -r f; do
  [[ -z "$f" ]] && continue
  v="$(version_of "$f")"
  if [[ -n "$max_base" ]] && ! [[ "$v" > "$max_base" ]]; then
    echo "❌ non-linear migration: ${f} (version ${v}) is not newer than the latest"
    echo "   migration already on ${BASE} (${max_base})."
    echo "   Atlas applies migrations in version order and prod has applied through"
    echo "   ${max_base}, so this file would land before already-applied migrations and"
    echo "   the deploy will fail. Rebase onto an up-to-date ${BASE} and regenerate the"
    echo "   migration (e.g. 'task make-migrations') so its timestamp is greater than ${max_base}."
    status=1
  fi
done < <(comm -13 <(base_migrations) <(disk_migrations))

if [[ "${status}" -eq 0 ]]; then
  echo "✅ Migrations are linear against ${BASE} (latest base version: ${max_base:-none})."
fi
exit "${status}"

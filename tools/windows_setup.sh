#!/bin/bash
set -euo pipefail

# The Docker apt repo below is Ubuntu-only; stop early on anything else.
if ! grep -q '^ID=ubuntu$' /etc/os-release; then
  echo "This script supports Ubuntu on WSL only. Install it with: wsl --install -d Ubuntu-24.04" >&2
  exit 1
fi

# --- bashrc env block ---
if ! grep -q '# windows-setup env' ~/.bashrc 2>/dev/null; then
  echo '
# windows-setup env
export GH_TELEMETRY=false
export DO_NOT_TRACK=true
export DISABLE_TELEMETRY=1
export OTEL_SDK_DISABLED=true
export TELEMETRY_DISABLED=true
export ATLAS_NO_ANON_TELEMETRY=true
export ATLAS_NO_UPDATE_NOTIFIER=true
export PATH=${PATH}:~/.local/bin' >> ~/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

# --- base packages ---
sudo apt -y update
sudo DEBIAN_FRONTEND=noninteractive apt -y install \
  git git-lfs libpq-dev gh curl jq build-essential ca-certificates
# --- git-lfs ---
echo ___ Configuring git-lfs:
# Register the git-lfs filters in ~/.gitconfig; the apt package alone doesn't do this.
# Without them, LFS-tracked files check out as ~130-byte pointer stubs instead of content.
git lfs install

# Registering the filters only affects *future* checkouts. If the repo was cloned before
# git-lfs was installed -- the normal case, since this script ships inside the repo -- the
# sample data warehouses (src/xngin/apiserver/testdata/*.zst) are still pointers on disk.
# `task test` and `task start` then fail when bootstrap-dwh-database tries to load them.
# `git lfs pull` backfills the real objects and is a safe no-op once they're present.
#
# Assumption: this script is run from a checkout (it lives in evidential-be/tools/). We
# resolve the repo from the script's own path so the caller's cwd doesn't matter. If it
# isn't in a work tree -- e.g. piped straight from curl -- we skip rather than fail.
lfs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")"
if repo_root="$(git -C "$lfs_dir" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "Fetching LFS objects in ${repo_root}"
  # Warn rather than abort: a transient fetch failure shouldn't kill the whole setup run.
  if git -C "$repo_root" lfs pull; then
    # '*' in the second column means the real object is on disk; '-' means still a pointer.
    git -C "$repo_root" lfs ls-files -s
  else
    echo "⚠️ 'git lfs pull' failed (network or auth?)."
    echo "   Re-run it in ${repo_root} before 'task test'."
  fi
else
  echo "⚠️ Not inside a git work tree -- skipping 'git lfs pull'."
  echo "   After cloning evidential-be, run 'git lfs pull' inside it."
fi

# --- Docker CE from Docker's official apt repo ---
echo ___ Installing Docker CE from official repo:
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# --- WSL: enable systemd (required so Docker starts as a service) ---
echo ___ Configuring WSL systemd:

if sudo grep -qs '^[[:space:]]*systemd[[:space:]]*=' /etc/wsl.conf; then
  # A systemd key is already there — force it to true, leaving every other line alone.
  sudo sed -i 's/^[[:space:]]*systemd[[:space:]]*=.*/systemd=true/' /etc/wsl.conf
elif sudo grep -qs '^[[:space:]]*\[boot\]' /etc/wsl.conf; then
  # [boot] exists but has no systemd key; insert one under the existing header
  # rather than appending a second [boot] section.
  sudo sed -i '/^[[:space:]]*\[boot\]/a systemd=true' /etc/wsl.conf
else
  # No [boot] section at all (or no file yet) — append one, preserving anything present.
  sudo tee -a /etc/wsl.conf > /dev/null <<'EOF'

[boot]
systemd=true
EOF
fi

# --- Add user to docker group ---
echo ___ Adding user to docker group:
sudo usermod -aG docker "$USER"

# --- language / tool installs ---
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv tool install prek
~/.local/bin/uv tool install go-task-bin
curl -sSf https://atlasgo.sh | sh -s -- -y
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.5/install.sh | bash
# --- Node 26 via nvm ---
echo ___ Installing Node 26:
export NVM_DIR="$HOME/.nvm"
set +e
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
set -e
nvm install 26
nvm alias default 26
npm install -g pnpm

echo ___ Done installing tools:
atlas version
~/.local/bin/uv --version
task --version


echo ___ Done. Now run 'wsl --shutdown' from Windows PowerShell, then reopen WSL so systemd + docker group membership take effect.

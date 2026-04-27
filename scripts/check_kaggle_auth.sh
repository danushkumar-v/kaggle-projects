#!/usr/bin/env bash
set -e

# Include user bin in PATH so jq is found when installed there
export PATH="$HOME/bin:$PATH"

echo "Checking Kaggle CLI auth..."

if ! command -v kaggle &> /dev/null; then
  echo "❌ kaggle CLI not installed. Run: pip install kaggle"
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "❌ ~/.kaggle/kaggle.json not found."
  echo "   1. Get token from https://www.kaggle.com/settings"
  echo "   2. Place at ~/.kaggle/kaggle.json"
  echo "   3. chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

# Permission check
perms=$(stat -c '%a' "$HOME/.kaggle/kaggle.json" 2>/dev/null || stat -f '%Lp' "$HOME/.kaggle/kaggle.json")
if [ "$perms" != "600" ]; then
  echo "⚠️  ~/.kaggle/kaggle.json permissions are $perms (should be 600)"
  echo "   Run: chmod 600 ~/.kaggle/kaggle.json"
fi

# Try a real API call
if kaggle kernels list --user "$(jq -r .username < $HOME/.kaggle/kaggle.json)" --page-size 1 &> /dev/null; then
  echo "✅ Kaggle CLI authenticated as $(jq -r .username < $HOME/.kaggle/kaggle.json)"
else
  echo "❌ Kaggle CLI auth failed. Check token at https://www.kaggle.com/settings"
  exit 1
fi

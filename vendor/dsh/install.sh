#!/usr/bin/env bash
set -euo pipefail
PREFIX=${1:-$HOME/.refinx-node/global}
npm install --global --prefix "$PREFIX" @deepseek-ai/dsh@0.2.0-rc.2
echo "installed $PREFIX/bin/dsh"
echo "add to PATH: export PATH=$PREFIX/bin:\$PATH"

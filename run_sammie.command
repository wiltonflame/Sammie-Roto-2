#!/usr/bin/env bash
# Change to script directory
cd "$(dirname "$0")" || { echo "Failed to change to script directory"; exit 1; }

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run install_dependencies.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Disable torch.compile for CorridorKey to prevent segfault with Qt on Linux
export CORRIDORKEY_OPT_MODE=lowvram

# Run the application
python3 launcher.py "$@"

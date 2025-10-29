#!/bin/bash

# >>> ./run_sage.sh pd.py

if [ $# -eq 0 ]; then
  echo "Usage: $0 <file_to_run.py>"
  exit 1
fi

FILE="$1"

sage << EOF
load("$FILE")
EOF

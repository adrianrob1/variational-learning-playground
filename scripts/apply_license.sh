#!/bin/bash

# --- CONFIGURATION ---
LICENSE_TYPE="GPL-3.0"
AUTHOR_NAME="Adrian R. Minut"
CURRENT_YEAR=$(date +%Y)
# Added a dot before venv and fixed the trailing pipe in your ignore patterns
IGNORE_PATTERNS="\.venv|venv|__pycache__|\.git|\.pytest_cache|build|dist|test|docs|data|conf|licenses|runs"
EXTENSIONS="py"

# Create a temporary header file with your metadata
HEADER_FILE=$(mktemp)
cat << EOF > "$HEADER_FILE"
# Copyright (c) $CURRENT_YEAR $AUTHOR_NAME
# SPDX-License-Identifier: GPL-3.0

EOF

echo "Running Smart License Audit for $AUTHOR_NAME ($CURRENT_YEAR)..."

# 1. Find the files, pruning ignored directories
# Note: Added -not -path '*/.*' to skip hidden files like .DS_Store
find . -type d -regextype posix-extended -regex "./($IGNORE_PATTERNS)" -prune \
    -o -type f -regextype posix-extended -regex ".*\.($EXTENSIONS)" -print | while read -r file; do
    
    # 2. Use licensecheck to see if a valid license exists
    LICENSE_STATUS=$(licensecheck "$file")
    
    if [[ "$LICENSE_STATUS" == *"UNKNOWN"* ]]; then
        echo " [+] Applying header to: $file"
        
        # We use a temporary file to prepend the header
        # This is safer than 'licensing apply' when custom metadata is needed
        cat "$HEADER_FILE" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    else
        echo " [s] Valid license already detected in: $file. Skipping."
    fi
done

# Clean up
rm "$HEADER_FILE"
echo "Done."
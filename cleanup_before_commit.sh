#!/usr/bin/env bash

# Friday AI - Pre-commit cleanup script
# Usage: ./cleanup_before_commit.sh

set -e

echo "🧹 Cleaning up Friday AI codebase for commit..."
echo "=============================================="

# Remove Python cache files
echo "🐍 Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove macOS system files
echo "🍎 Removing macOS system files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name ".AppleDouble" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".LSOverride" -delete 2>/dev/null || true

# Remove temporary files
echo "🗑️  Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.bak" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

# Remove empty Python files (except __init__.py)
echo "📝 Checking for empty Python files..."
find . -name "*.py" -size 0 -not -name "__init__.py" | while read -r file; do
    echo "   Found empty file: $file"
done

# Check for large files that shouldn't be committed
echo "📦 Checking for large files..."
find . -type f -size +50M | grep -v ".git" | while read -r file; do
    echo "   ⚠️  Large file detected: $file ($(du -h "$file" | cut -f1))"
done

# Show git status
echo "📊 Current git status:"
git status --porcelain | head -20

echo ""
echo "✅ Cleanup completed!"
echo "💡 Remember to:"
echo "   - Review changes with: git diff"
echo "   - Add files with: git add ."
echo "   - Commit with: git commit -m 'your message'"
echo "   - Push with: git push origin $(git branch --show-current)"

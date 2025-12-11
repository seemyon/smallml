#!/bin/bash
#
# SmallML Package Build Script
# =============================
#
# This script builds the SmallML package for PyPI distribution.
#
# Usage:
#   bash build_package.sh
#

echo "=============================================================="
echo " Building SmallML Package"
echo "=============================================================="
echo ""

# Check if build tools are installed
echo "Checking for build tools..."
python -c "import build" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing build tools..."
    pip install build twine
fi
echo "✓ Build tools ready"
echo ""

# Clean previous builds
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info/
echo "✓ Cleaned"
echo ""

# Build package
echo "Building package..."
echo "-" "-----------------------------------------------------------------"
python -m build
echo "-" "-----------------------------------------------------------------"
echo ""

# Check if build succeeded
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ ERROR: Build failed - no dist/ directory or empty"
    exit 1
fi

echo "✓ Build complete!"
echo ""

# Show what was built
echo "Build artifacts:"
ls -lh dist/
echo ""

# Verify package contents
echo "Verifying package contents..."
echo ""
echo "Contents of wheel:"
echo "-" "-----------------------------------------------------------------"
unzip -l dist/*.whl | grep smallml | head -20
echo "..."
echo "-" "-----------------------------------------------------------------"
echo ""

echo "Contents of source distribution:"
echo "-" "-----------------------------------------------------------------"
tar -tzf dist/*.tar.gz | grep smallml | head -20
echo "..."
echo "-" "-----------------------------------------------------------------"
echo ""

echo "=============================================================="
echo " ✓ Package Built Successfully"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  1. Test installation: bash test_installation.sh"
echo "  2. Upload to TestPyPI: twine upload --repository testpypi dist/*"
echo "  3. Upload to PyPI: twine upload dist/*"
echo ""

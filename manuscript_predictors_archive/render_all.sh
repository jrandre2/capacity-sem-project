#!/bin/bash
# Render manuscript in all formats
# Usage: ./render_all.sh
# Set CAPACITY_SEM_SKIP_PIPELINE=1 to skip running the pipeline

set -e

cd "$(dirname "$0")"

echo "================================"
echo "Rendering Predictors Manuscript"
echo "================================"

# Run the predictor discovery analysis unless skipped
if [ -z "$CAPACITY_SEM_SKIP_PIPELINE" ]; then
    echo "Running predictor discovery analysis..."
    cd ..
    python run_predictor_discovery.py
    cd manuscript_predictors
else
    echo "Skipping pipeline (CAPACITY_SEM_SKIP_PIPELINE is set)"
fi

# Render all formats
echo ""
echo "Rendering HTML..."
quarto render index.qmd --to html

echo ""
echo "Rendering PDF..."
quarto render index.qmd --to pdf

echo ""
echo "Rendering DOCX..."
quarto render index.qmd --to docx

echo ""
echo "================================"
echo "Rendering complete!"
echo "Outputs in: _output/"
echo "================================"

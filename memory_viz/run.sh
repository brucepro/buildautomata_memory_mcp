#!/bin/bash
# Run Memory Graph Visualization Server

echo "Starting Memory Graph Visualization..."
echo "Open http://localhost:8686 in your browser"
echo ""

cd "$(dirname "$0")"
python3 app.py

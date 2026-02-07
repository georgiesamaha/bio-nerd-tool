#!/bin/bash

# gget-MCP Local Setup Script
# This script sets up Ollama with Qwen2.5-Coder-3B and configures the gget-MCP server

echo "=== Setting up gget-MCP with Local Model ==="

# 1. Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download model  
echo "Downloading Qwen2.5-Coder-3B model (~2GB)..."
ollama pull qwen2.5-coder:3b

# 3. Test Ollama works
echo "Testing Ollama installation..."
ollama run qwen2.5-coder:3b "What is a gene?"

# 4. Install gget-MCP server dependencies
echo "Installing gget-MCP dependencies..."
pip install -e .

# 6. Test functionality  
echo "Testing functionality..."
echo "1. Testing basic Python..."
python3 -c "print('✅ Python works')"

echo "2. Testing offline gene lookup..."
python3 offline_gget.py TP53 || echo "⚠️  Offline mode test"

echo "3. Testing simple gene info..."  
python3 simple_tp53.py || echo "⚠️  Simple test issue"

# 6. Start Ollama service in background
echo "Starting Ollama service..."
ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama service started (PID: $OLLAMA_PID)"

echo "\n=== Setup Complete ==="
echo "Ollama API available at: http://localhost:11434"
echo "\\n🧬 GENE QUERY TOOLS:"
echo "  • Offline mode:  python3 offline_gget.py TP53"  
echo "  • Network mode:  python3 gget_cli.py TP53"
echo "  • Simple test:   python3 simple_tp53.py"
echo "\\n📚 CONFIGURATION:"
echo "  • Claude Desktop: config/claude_desktop_config.json"
echo "  • LM Studio: config/lm_studio_config.json" 
echo "  • Custom API: config/ollama_config.py"
echo "\\n🔧 TROUBLESHOOTING:"
echo "  • If network errors → Use offline_gget.py"
echo "  • If import errors → Run 'pip install -e .'"
echo "  • See TROUBLEHOOTING.md for details"
echo "\\nTo stop Ollama: kill $OLLAMA_PID"

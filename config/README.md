# Configuration Files for gget-MCP + Local Model Integration

This directory contains configuration files for integrating the gget-MCP server with your local Qwen2.5-Coder-3B model running on Ollama.

## Files Overview

### `claude_desktop_config.json`
Configuration for Claude Desktop app to use gget-MCP with local model.

**Setup Instructions:**
1. Copy this file to: `~/.config/claude/claude_desktop_config.json`
2. Update the `PYTHONPATH` to match your gget-mcp installation directory
3. Restart Claude Desktop

### `lm_studio_config.json` 
Configuration for LM Studio to use gget-MCP server.

**Setup Instructions:**
1. Open LM Studio
2. Go to Settings → MCP Servers
3. Import this configuration
4. Make sure model path matches your installation

### `ollama_config.py`
Python configuration for custom clients using OpenAI-compatible API.

**Usage:**
```python
from config.ollama_config import OPENAI_CONFIG, MCP_CONFIG

# Use with OpenAI client
import openai
client = openai.OpenAI(**OPENAI_CONFIG)

# Use with MCP
subprocess.run([MCP_CONFIG["command"]] + MCP_CONFIG["args"])
```

## Quick Start

1. **Start Ollama service:**
   ```bash
   ollama serve
   ```

2. **Verify model is available:**
   ```bash
   ollama list | grep qwen2.5-coder:3b
   ```

3. **Test integration:**
   ```bash
   python test_integration.py
   ```

4. **Choose your client and copy the appropriate config file**

## Client-Specific Setup

### For Claude Desktop

1. **Find config directory:**
   ```bash
   # Linux/macOS
   mkdir -p ~/.config/claude
   ```

2. **Copy configuration:**
   ```bash
   cp config/claude_desktop_config.json ~/.config/claude/claude_desktop_config.json
   ```

3. **Update paths in the config file to match your setup**

4. **Restart Claude Desktop**

### For LM Studio

1. **Open LM Studio**
2. **Go to: Chat → Settings → MCP**
3. **Click "Import Configuration"**
4. **Select `lm_studio_config.json`**
5. **Verify paths and start chatting**

### For Custom Applications

Use the `ollama_config.py` file as a reference for:
- API endpoints
- Model names  
- Environment variables
- MCP server startup commands

## Example Usage

Once configured, you can ask your AI assistant:

- **"What is the TP53 gene?"**
- **"Use gget_info to get information about ENSG00000157764"**
- **"Tell me about the BRAF gene and include confidence levels"**

### Expected Response Format

```json
{
  "success": true,
  "data": {
    "gene_name": "TP53",
    "description": "tumor protein p53",
    "biotype": "protein_coding",
    "chromosome": "17"
  },
  "epistemic_metadata": {
    "confidence_level": "high",
    "confidence_score": 0.95,
    "uncertainty_sources": []
  },
  "provenance_metadata": {
    "primary_source": "Ensembl",
    "data_version": "110",
    "query_timestamp": "2026-02-07T10:30:00Z"
  }
}
```

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check model is installed
ollama list
```

### MCP Server Issues
```bash
# Test server directly
python -m gget_mcp.server

# Check dependencies
pip install -e .

# Run integration tests
python test_integration.py
```

### Permission Issues
```bash
# Fix Python path
export PYTHONPATH="/home/ubuntu/gget-mcp:$PYTHONPATH"

# Fix file permissions
chmod +x test_integration.py
chmod +x download.sh
```

## Environment Variables

Set these for optimal operation:

```bash
# Required
export PYTHONPATH="/home/ubuntu/gget-mcp"

# Optional tuning
export GGET_MCP_LOG_LEVEL="INFO"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

## Security Notes

- All data queries go through public bioinformatics databases only
- No private data access or modification capabilities
- Rate limiting prevents abuse
- Complete audit logging for accountability
- Local model means no data sent to external APIs

## Performance Tips

- **RAM**: 8GB+ recommended for smooth operation
- **Storage**: Keep 4GB+ free for model cache
- **CPU**: Multi-core helpful for concurrent queries
- **Network**: Only needed for initial downloads, then fully offline
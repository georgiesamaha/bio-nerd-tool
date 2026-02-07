# gget-MCP: Safety-Bounded Bioinformatics Query Server

A Model Context Protocol (MCP) server that provides access to gget bioinformatics tools with comprehensive AI safety boundaries, epistemic controls, and failure mode management.

## Features

### Core Functionality
- **Gene Information Queries**: Access comprehensive gene/transcript/protein information via gget.info
- **Structured Outputs**: All responses use validated Pydantic schemas
- **Error Boundaries**: Comprehensive error handling with explicit unknown states

### AI Safety Framework
- **Authority Boundaries**: Clear limits on server capabilities and scope
- **Epistemic Stance**: Confidence levels and uncertainty quantification
- **Failure Mode Management**: Graceful degradation with informative error states
- **Motivation Model**: Accuracy over completeness, evidence over helpfulness
- **Provenance Tracking**: Full traceability of data sources and processing steps
- **Refusal Templates**: Pre-written responses for out-of-scope queries

### Safety Controls
- **Input Validation**: Strict schema validation for all inputs
- **Output Validation**: Guaranteed response structure with safety metadata
- **Rate Limiting**: Built-in query throttling to prevent abuse
- **Audit Logging**: Complete request/response logging for accountability

## Documentation

- **[Complete Building Guide](docs/BUILDING_GUIDE.md)**: Comprehensive step-by-step documentation of the entire development process, including design decisions, safety framework implementation, local model setup, and beginner-friendly tutorials
- **[Local Model Setup](docs/BUILDING_GUIDE.md#local-model-setup)**: Instructions for cost-effective deployment with Qwen2.5-Coder-3B
- **[AI Safety Framework](docs/BUILDING_GUIDE.md#ai-safety-framework)**: Detailed explanation of safety controls and boundaries

## Installation

### Quick Setup with Local Model

```bash
# Run the automated setup script
chmod +x download.sh
./download.sh
```

This script will:
- Install Ollama 
- Download Qwen2.5-Coder-3B model (~2GB)
- Install gget-MCP dependencies
- Test the integration
- Start Ollama service

### Manual Installation

```bash
# Install dependencies
pip install -e .

# Install Ollama separately
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:3b

# Run the server
python -m gget_mcp.server
```

### Verify Setup

```bash
# Test the complete integration
python test_integration.py
# Quick test after confidence fix
python3 quick_test.py

# Use the command-line tool
python3 gget_cli.py TP53```

## Usage with MCP Clients

### With Local Model (Recommended)

After running `./download.sh`, configure your MCP client:

**Claude Desktop:**
```bash
# Copy configuration 
cp config/claude_desktop_config.json ~/.config/claude/claude_desktop_config.json
# Restart Claude Desktop
```

**LM Studio:**  
Import `config/lm_studio_config.json` in LM Studio's MCP settings.

**Custom Client:**
Use `config/ollama_config.py` for OpenAI-compatible API integration.

### Generic MCP Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "gget-mcp": {
      "command": "python",
      "args": ["-m", "gget_mcp.server"],
      "env": {
        "PYTHONPATH": "/home/ubuntu/gget-mcp"
      }
    }
  }
}
```

### Local Model Benefits

- **Cost-effective**: No API fees
- **Privacy**: Data stays local  
- **Reliability**: No external dependencies
- **Performance**: ~2GB model optimized for bioinformatics

## Available Tools

### `gget_info`
Retrieve comprehensive gene/transcript/protein information.

**Input Schema**:
```json
{
  "gene_id": "ENSG00000157764",  // Required: Gene identifier
  "confidence_level": "standard"  // Optional: low|standard|high
}
```

**Output Schema**: Structured gene information with provenance and confidence metadata.

## Command-Line Interface

**🔌 Offline Mode (Works Immediately):**
```bash
# Works without network - cached gene data
python3 offline_gget.py TP53
python3 offline_gget.py BRCA1
python3 offline_gget.py EGFR

# Get help and see available genes
python3 offline_gget.py --help
```

**🌐 Network Mode (Full Database Access):**
```bash
# Requires internet for live data
python3 gget_cli.py TP53
python3 gget_cli.py BRCA1 
python3 gget_cli.py ENSG00000157764

# Specify confidence level
python3 gget_cli.py TP53 high
```

**⚡ Quick Test:**
```bash
# Simple test that always works
python3 simple_tp53.py
```

**Features:**
- **Offline Mode**: Works instantly with cached data for common genes (TP53, BRCA1, EGFR, MYC, BRCA2)
- **Network Mode**: Full access to live Ensembl/NCBI databases (requires internet)
- **Safety boundaries**: Confidence levels and error handling
- **Multiple formats**: Gene symbols, Ensembl IDs, aliases

**Troubleshooting:**
- If `gget_cli.py` hangs → Use `offline_gget.py` (works without internet)
- If network errors → Check internet connection or use offline mode
- If import errors → Run `pip install -e .`

## Safety Guarantees

1. **No Hallucinations**: All data sourced directly from Ensembl/NCBI databases
2. **Explicit Unknowns**: Clear indication when information is unavailable
3. **Traceable Sources**: Every fact includes source attribution
4. **Bounded Scope**: Server only operates within defined bioinformatics domain
5. **Error Transparency**: All failures include diagnostic information

## System Limitations

- **Read-Only**: No data modification or external system interactions
- **Public Data Only**: No access to private or sensitive datasets
- **Rate Limited**: Maximum 60 queries per minute per client
- **Domain Bounded**: Strictly bioinformatics-focused, no general web access

## Architecture

```
gget-mcp/
├── gget_mcp/
│   ├── __init__.py
│   ├── server.py           # MCP server implementation
│   ├── safety/             # AI safety framework
│   │   ├── __init__.py
│   │   ├── boundaries.py   # Authority and domain boundaries
│   │   ├── epistemic.py    # Confidence and uncertainty management
│   │   ├── failures.py     # Error handling and refusal templates
│   │   └── provenance.py   # Source tracking and metadata
│   ├── tools/              # MCP tool implementations
│   │   ├── __init__.py
│   │   └── gget_info.py    # Gene information queries
│   └── schemas/            # Pydantic validation schemas
│       ├── __init__.py
│       ├── inputs.py       # Input validation schemas
│       └── outputs.py      # Output validation schemas
├── tests/
├── pyproject.toml
└── README.md
```

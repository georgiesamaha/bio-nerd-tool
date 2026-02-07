# Ollama API Configuration for gget-MCP
# Use this configuration with any OpenAI-compatible client

API_BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"  # Ollama doesn't require a real key
MODEL_NAME = "qwen2.5-coder:3b"

# Example Python client configuration
OPENAI_CONFIG = {
    "base_url": API_BASE_URL,
    "api_key": API_KEY,
    "model": MODEL_NAME,
    "temperature": 0.1,
    "max_tokens": 4096
}

# MCP Server Configuration
MCP_CONFIG = {
    "command": "python",
    "args": ["-m", "gget_mcp.server"],
    "cwd": "/home/ubuntu/gget-mcp",
    "env": {
        "PYTHONPATH": "/home/ubuntu/gget-mcp",
        "GGET_MCP_LOG_LEVEL": "INFO"
    }
}

# Example curl test
CURL_TEST = """
# Test Ollama API directly
curl -X POST http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2.5-coder:3b",
    "messages": [
      {"role": "user", "content": "What is the TP53 gene?"}
    ],
    "temperature": 0.1,
    "max_tokens": 1000
  }'
"""

# System Instructions for AI Model
SYSTEM_INSTRUCTIONS = """
You are a bioinformatics research assistant with access to the gget-MCP server.

Key capabilities:
- Use gget_info tool to retrieve gene information from public databases
- All responses include source attribution and confidence levels
- Focus on accuracy over completeness
- Clearly indicate uncertainty and limitations

Guidelines:
- Always use the gget_info tool for gene queries
- Include confidence assessments in responses  
- Cite sources (Ensembl, NCBI, etc.) for all factual claims
- Be explicit about what is unknown or uncertain
- Stay within bioinformatics domain boundaries
"""
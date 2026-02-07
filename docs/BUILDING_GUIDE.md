# Building a Safety-Bounded MCP Server: Complete Guide

This guide walks through the complete process of building the gget-MCP server from scratch, including all safety features and local model setup.

## Overview

We built a Model Context Protocol (MCP) server that provides access to bioinformatics data through the `gget` library while implementing comprehensive AI safety controls. The system prioritises accuracy over completeness, evidence over helpfulness, and silence over speculation.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Project Structure](#project-structure)
3. [Implementation Process](#implementation-process)
4. [AI Safety Framework](#ai-safety-framework)
5. [Local Model Setup](#local-model-setup)
6. [Testing and Validation](#testing-and-validation)
7. [Usage Examples](#usage-examples)

## Design Philosophy

### Core Principles

**Safety First**: Every component was designed with AI safety as the primary concern

- **Authority boundaries**: Clear limits on what the system can and cannot do
- **Epistemic stance**: Explicit confidence levels and uncertainty quantification  
- **Failure mode management**: Graceful degradation with informative error states
- **Provenance tracking**: Complete audit trail of all data sources
- **Motivation model**: Optimise for accuracy over completeness

### Safety Guarantees

1. **No hallucinations**: All data comes directly from authoritative sources
2. **Explicit unknowns**: Clear indication when information is unavailable
3. **Traceable sources**: Every fact includes source attribution
4. **Bounded scope**: System only operates within defined bioinformatics domain
5. **Error transparency**: All failures include diagnostic information

## Project Structure

```
gget-mcp/
├── gget_mcp/
│   ├── __init__.py              # Package initialisation
│   ├── server.py                # Main MCP server implementation
│   ├── safety/                  # AI safety framework
│   │   ├── __init__.py
│   │   ├── boundaries.py        # Authority and domain boundaries
│   │   ├── epistemic.py         # Confidence and uncertainty management
│   │   ├── failures.py          # Error handling and refusal templates
│   │   └── provenance.py        # Source tracking and metadata
│   ├── tools/                   # MCP tool implementations
│   │   ├── __init__.py
│   │   └── gget_info.py         # Gene information queries
│   └── schemas/                 # Pydantic validation schemas
│       ├── __init__.py
│       ├── inputs.py            # Input validation schemas
│       └── outputs.py           # Output validation schemas
├── tests/                       # Test suite
├── docs/                        # Documentation
├── pyproject.toml              # Project configuration
└── README.md                   # Project overview
```

## Implementation Process

### Step 1: Project Initialisation

First, we set up the basic project structure and dependencies:

```bash
# Create project directory
mkdir gget-mcp
cd gget-mcp

# Create package structure
mkdir -p gget_mcp/safety gget_mcp/tools gget_mcp/schemas tests docs
```

#### Key dependencies (`pyproject.toml`)

- `mcp>=1.2.0` - Model Context Protocol implementation
- `gget>=0.28.0` - Bioinformatics query library
- `pydantic>=2.0.0` - Data validation and settings management
- `httpx>=0.24.0` - HTTP client for external requests

### Step 2: Safety Framework Implementation

#### Authority Boundaries (`safety/boundaries.py`)

We implemented strict authority boundaries to define what the system can and cannot do:

```python
class AuthorityBoundaries:
    """Defines the authority and capability boundaries of the server."""
    
    allowed_domains: Set[DomainScope] = {
        DomainScope.BIOINFORMATICS, 
        DomainScope.GENE_INFORMATION, 
        DomainScope.PUBLIC_DATABASES
    }
    
    prohibited_actions: Set[ProhibitedAction] = {
        ProhibitedAction.DATA_MODIFICATION,
        ProhibitedAction.EXTERNAL_API_CALLS,
        ProhibitedAction.ARBITRARY_CODE_EXECUTION,
        # ... more restrictions
    }
```

**Key Features**:
- Enum-based operation types for type safety
- Rate limiting (60 queries/minute)
- Response size limits
- Clear violation messages

#### Epistemic Controls (`safety/epistemic.py`)

Implemented confidence assessment and uncertainty quantification:

```python
class ConfidenceLevel(str, Enum):
    HIGH = "high"           # >95% confidence
    STANDARD = "standard"   # 80-95% confidence  
    LOW = "low"            # 50-80% confidence
    UNCERTAIN = "uncertain" # <50% confidence
    UNKNOWN = "unknown"     # No reliable information
```

**Assessment Function**:
```python
def assess_gene_info_confidence(gene_data: Dict[str, Any]) -> EpistemicState:
    """Assess confidence based on data completeness and quality."""
    # Check data completeness
    # Assess annotation quality
    # Determine confidence level
    # Build uncertainty quantification
```

#### Failure Mode Management (`safety/failures.py`)

Created comprehensive error handling with refusal templates:

```python
class RefusalTemplate:
    """Template for refusing requests with clear explanation."""
    
    reason: RefusalReason
    message: str  # Clear, non-apologetic explanation
    alternative_suggestions: List[str]
    boundary_explanation: Optional[str]
```

**Pre-configured Refusals**:
- Out of scope requests
- Insufficient confidence data
- Prohibited actions  
- Invalid inputs
- Rate limiting
- Safety violations

#### Provenance Tracking (`safety/provenance.py`)

Implemented complete source attribution and data lineage:

```python
class ProvenanceRecord:
    """Complete provenance record for a response."""
    
    query_fingerprint: str
    data_lineages: List[DataLineage] 
    system_version: str
    processing_environment: Dict[str, str]
    audit_trail: List[str]
```

**Source Registry**:
- Ensembl (reliability: 0.95)
- NCBI (reliability: 0.95) 
- gget library (reliability: 0.85)
- Full citations and license information

### Step 3: Input/Output Validation

#### Input Schemas (`schemas/inputs.py`)

Created strict input validation with security controls:

```python
class GgetInfoInput(BaseModel):
    gene_id: str = Field(min_length=1, max_length=50)
    confidence_level: ConfidenceLevel = ConfidenceLevel.STANDARD
    include_sequences: bool = False
    species: Optional[str] = None
    
    @validator('gene_id')
    def validate_gene_id_format(cls, v):
        # Remove dangerous characters
        # Validate against known patterns
        # Security sanitisation
```

**Validation Functions**:
- `validate_gene_identifier()` - Supports Ensembl, NCBI, UniProt formats
- `check_input_safety()` - Additional security checks
- Pattern matching for scientific names

#### Output Schemas (`schemas/outputs.py`)

Structured responses with full metadata:

```python
class GgetInfoOutput(BaseModel):
    success: bool
    data: Optional[GgetInfoResponseData]
    epistemic_metadata: EpistemicMetadata  # Confidence info
    provenance_metadata: ProvenanceMetadata  # Source attribution
    safety_flags: List[str]
    recommendations: List[str]
```

### Step 4: Tool Implementation

#### gget.info Tool (`tools/gget_info.py`)

Implemented the main bioinformatics query tool with safety controls:

```python
class GgetInfoTool:
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Input validation and sanitisation
        # 2. Safety and boundary checks  
        # 3. Execute gget query with error handling
        # 4. Assess confidence and create epistemic state
        # 5. Check confidence requirements
        # 6. Create provenance record
        # 7. Structure response data
        # 8. Create safe response with metadata
```

**Safety Checks**:
- Authority boundary validation
- Domain boundary validation  
- System invariant validation
- Input safety assessment

### Step 5: MCP Server Implementation

#### Main Server (`server.py`)

Created the FastMCP server with comprehensive safety integration:

```python
class GgetMCPServer:
    def __init__(self):
        # Configure safety logging
        # Initialise safety systems
        # Initialise tools with safety controls
        # Register MCP tools
        
    def _register_tools(self):
        @self.app.tool()
        async def gget_info(...):
            # Execute with full safety controls
            
        @self.app.tool() 
        async def system_capabilities():
            # Expose system limitations
            
        @self.app.tool()
        async def system_status():
            # Health and status information
```

**System Instructions**:
Comprehensive instructions that define the system's identity, capabilities, limitations, and safety principles.

## AI Safety Framework

### 1. Authority Boundaries
- **Domain Scope**: Bioinformatics, gene information, public databases only
- **Operations**: Read-only queries exclusively
- **Rate Limits**: 60 queries per minute per client
- **Response Size**: Maximum 10,000 characters

### 2. Epistemic Controls
- **Confidence Scoring**: 0.0-1.0 numerical confidence
- **Evidence Quality**: Peer-reviewed, database-verified, computational prediction levels
- **Uncertainty Types**: Measurement error, incomplete data, conflicting sources, etc.
- **Refusal Thresholds**: Refuse queries below 50% confidence

### 3. Failure Mode Management
- **Graceful Degradation**: Informative error messages instead of crashes
- **Refusal Templates**: Pre-written, non-apologetic refusal messages
- **Recovery Suggestions**: Actionable alternatives when requests fail
- **Error Classification**: Categorised by severity and recovery options

### 4. Provenance Tracking
- **Source Attribution**: Every fact traceable to authoritative source
- **Data Lineage**: Complete processing pipeline documentation
- **Integrity Verification**: Cryptographic checksums for data integrity
- **Audit Trail**: Full logs for accountability

### 5. System Invariants
Unchangeable rules that ensure safety:
- Read-only operation mode
- No external API calls beyond gget's mechanisms
- Public data only
- Traceable sources required
- Bounded scope enforcement

## Local Model Setup

### Why Use Local Models?

- **Cost Control**: No API fees for queries
- **Privacy**: Data stays on your machine
- **Reliability**: No dependency on external services
- **Customisation**: Full control over model behavior

### Recommended Model: Qwen2.5-Coder-3B-Instruct

**Why This Model**:
- **Size**: ~2GB (manageable storage)
- **Performance**: Excellent with structured data and APIs
- **Instruction Following**: Reliable adherence to safety constraints
- **Bioinformatics Aware**: Understands scientific terminology

### Installation Steps

#### Option 1: Ollama (Recommended)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download model (auto-downloads ~2GB)
ollama pull qwen2.5-coder:3b

# 3. Start Ollama service
ollama serve

# 4. Test the model
ollama run qwen2.5-coder:3b "What is a gene?"
```

**Ollama Benefits**:
- Automatic quantisation and optimisation
- Built-in API server on `localhost:11434`
- GPU acceleration when available
- Easy model management

#### Option 2: LM Studio (GUI)

1. **Download**: Visit https://lmstudio.ai/
2. **Install**: Follow platform-specific instructions  
3. **Search**: Look for "qwen2.5-coder-3b"
4. **Download**: Select quantised version (Q4_K_M recommended)
5. **Load**: Load model in chat interface
6. **API**: Enable local API server

#### Option 3: Manual Setup

```bash
# Install dependencies
pip install transformers torch

# Download model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-Coder-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype='auto',
    device_map='auto'
)
print('Model downloaded successfully')
"
```

### MCP Client Configuration

#### For Claude Desktop

Create or edit `~/.config/claude/claude_desktop_config.json`:

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

#### For LM Studio

In LM Studio settings:

```json
{
  "mcp_servers": {
    "gget-mcp": {
      "command": "python",
      "args": ["-m", "gget_mcp.server"],
      "type": "stdio"
    }
  }
}
```

#### For Custom Clients

Use the Ollama API endpoint:

```python
import requests

# Query the local model
response = requests.post('http://localhost:11434/v1/chat/completions', json={
    "model": "qwen2.5-coder:3b",
    "messages": [
        {"role": "user", "content": "Use the gget_info tool to get information about gene TP53"}
    ],
    "tools": [
        # Include MCP tool definitions
    ]
})
```

## Testing and Validation

### Setup Test Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run safety boundary tests specifically
pytest tests/test_safety_boundaries.py -v
```

### Manual Testing Steps

#### 1. Basic Functionality Test

```bash
# Start the server
python -m gget_mcp.server

# In another terminal, test with MCP client
# Or test individual components:
python -c "
from gget_mcp.tools.gget_info import GgetInfoTool
import asyncio

tool = GgetInfoTool()
result = asyncio.run(tool.execute({
    'gene_id': 'ENSG00000157764',
    'confidence_level': 'standard'
}))
print(result)
"
```

#### 2. Safety Boundary Tests

```python
# Test authority boundaries
from gget_mcp.safety.boundaries import DEFAULT_AUTHORITY_BOUNDARIES

# Should reject prohibited actions
assert boundaries.is_action_prohibited("data_modification")

# Should allow bioinformatics queries
assert boundaries.is_operation_allowed("read_only_query", "bioinformatics")
```

#### 3. Input Validation Tests

```python
# Test input sanitisation
from gget_mcp.schemas.inputs import validate_and_sanitize_input

# Valid input
valid_input = validate_and_sanitize_input({
    "gene_id": "ENSG00000157764",
    "confidence_level": "standard"
})

# Invalid input should raise exception
try:
    validate_and_sanitize_input({"gene_id": "<script>alert('xss')</script>"})
    assert False, "Should have rejected malicious input"
except ValueError:
    pass  # Expected
```

#### 4. Confidence Assessment Tests

```python
# Test epistemic controls
from gget_mcp.safety.epistemic import assess_gene_info_confidence

# Complete data should have high confidence
confidence = assess_gene_info_confidence({
    "gene_name": "TP53",
    "description": "tumor protein p53",
    "biotype": "protein_coding",
    "chromosome": "17"
})
assert confidence.confidence_level in ["standard", "high"]

# Empty data should have unknown confidence
confidence = assess_gene_info_confidence({})
assert confidence.confidence_level == "unknown"
```

### Integration Testing

#### Test MCP Tool Registration

```python
# Test that tools are properly registered
from gget_mcp.server import create_mcp_server

server = create_mcp_server()
# Verify tools are registered with proper schemas
```

#### Test End-to-End Flow

```bash
# 1. Start local model
ollama serve &

# 2. Start MCP server  
python -m gget_mcp.server &

# 3. Test with client
# Use your preferred MCP client to test queries
```

## Usage Examples

### Basic Gene Query

```python
# Query: "Get information about the TP53 gene"
# Tool call: gget_info
{
    "gene_id": "TP53",
    "confidence_level": "standard"
}

# Expected response includes:
# - Gene information with full provenance
# - Confidence assessment
# - Source attributions
# - Processing audit trail
```

### Error Handling Example

```python
# Query: "Get information about INVALID_GENE_123"
# Expected: Graceful failure with informative message

{
    "success": false,
    "error": {
        "failure_mode": {
            "category": "input_validation",
            "user_message": "The gene identifier provided is not valid or recognised",
            "recovery_suggestions": [
                "Check the gene ID format (e.g., ENSG00000157764 for Ensembl)",
                "Verify the gene exists in public databases"
            ]
        }
    }
}
```

### System Capabilities Query

```python
# Query: "What can this system do?"
# Tool call: system_capabilities

# Response includes:
# - Available tools
# - Supported databases  
# - Data types
# - Limitations
# - Safety measures
```

### Confidence-Based Refusal

```python
# Query for gene with insufficient data
# System response:

{
    "success": false,
    "error": {
        "refusal_template": {
            "reason": "insufficient_confidence", 
            "message": "Available data has insufficient confidence for a reliable response",
            "alternative_suggestions": [
                "Try a more specific gene identifier",
                "Check if newer data might be available"
            ]
        }
    }
}
```

## Performance Optimisation

### Local Model Performance

**Hardware Recommendations**:
- **RAM**: 8GB+ for smooth operation
- **Storage**: 4GB+ free space for model and cache
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but helpful (even integrated GPUs)

**Optimisation Settings**:
```bash
# For Ollama
export OLLAMA_NUM_PARALLEL=1  # Limit concurrent requests
export OLLAMA_MAX_LOADED_MODELS=1  # Conserve memory

# For systems with limited resources
ollama run qwen2.5-coder:3b --ctx-size 2048  # Reduce context window
```

### MCP Server Performance

**Configuration Tuning**:
```python
# In server configuration
RATE_LIMIT = 30  # Reduce for slower hardware
MAX_RESPONSE_SIZE = 5000  # Smaller responses
CONFIDENCE_THRESHOLD = 0.7  # Higher threshold for quality
```

## Troubleshooting

### Common Issues

#### 1. gget Import Errors
```bash
# Solution: Install gget
pip install gget>=0.28.0

# Verify installation
python -c "import gget; print(gget.__version__)"
```

#### 2. MCP Connection Issues
```bash
# Check MCP server is running
python -m gget_mcp.server

# Verify tool registration
# Look for "MCP tools registered successfully" in logs
```

#### 3. Local Model Memory Issues
```bash
# Reduce model size
ollama pull qwen2.5-coder:3b-q4_0  # Smaller quantisation

# Monitor memory usage
htop
```

#### 4. Permission Errors
```bash
# Fix permissions
chmod +x $(which python)
pip install --user gget-mcp-server
```

### Debug Mode

Enable detailed logging:

```python
# Set environment variable
export GGET_MCP_DEBUG=1

# Or modify server.py
logging.getLogger().setLevel(logging.DEBUG)
```

## Next Steps

### Extending Functionality

1. **Add More Tools**: Implement additional gget functions (sequences, enrichment, etc.)
2. **Custom Databases**: Add support for organisation-specific databases
3. **Batch Processing**: Implement batch queries with progress tracking
4. **Caching**: Add intelligent caching for frequently requested data

### Advanced Safety Features

1. **User Authentication**: Add user-specific rate limiting and access controls
2. **Query Auditing**: Enhanced audit trails for compliance
3. **Custom Confidence Models**: Train domain-specific confidence assessments
4. **Federated Learning**: Share anonymised usage patterns for improvement

### Production Deployment

1. **Docker Containerisation**: Package for easy deployment
2. **API Gateway**: Add authentication and load balancing  
3. **Monitoring**: Implement health checks and metrics
4. **High Availability**: Multi-instance deployment with failover

## Conclusion

This guide showed how to build a production-ready MCP server with comprehensive AI safety controls. The key innovations include:

- **Safety-First Design**: Every component prioritises reliability and transparency
- **Comprehensive Error Handling**: Graceful failures with actionable guidance
- **Complete Provenance**: Full audit trail from query to response
- **Epistemic Honesty**: Explicit confidence levels and uncertainty quantification
- **Local Model Integration**: Cost-effective deployment with privacy protection

The resulting system provides a robust foundation for safe AI-assisted bioinformatics research while maintaining the flexibility to extend and customise for specific use cases.
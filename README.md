# Bio-Nerd Tool: AI-Powered Bioinformatics Query Interface

An intelligent bioinformatics tool that combines the power of the `gget` library with local AI interpretation via Ollama. Built on the FastMCP framework, it provides natural language querying of gene databases with smart gene detection and comprehensive biological insights.

## Features

### 🧬 Bio-Nerd CLI Interface
- **Natural Language Queries**: Ask about genes in plain English - "tell me about TP53"
- **Smart Gene Detection**: Automatically identifies gene symbols and Ensembl IDs from text
- **Typo Filtering**: AI-powered validation prevents common English words from being treated as genes
- **Multiple Gene Formats**: Supports gene symbols (TP53, BRCA1), Ensembl IDs (ENSG00000141510), and aliases

### 🤖 AI-Powered Interpretation  
- **Local AI Integration**: Uses Ollama with llama3:8b for biological insights
- **No API Costs**: Completely local processing with no external dependencies
- **Biological Expertise**: Model fine-tuned understanding of genetics and molecular biology
- **Interactive Processing**: Real-time interpretation of gene functions and significance

### 🔬 Comprehensive Bioinformatics Data
- **gget Library Integration**: Direct access to Ensembl, NCBI, and other genomics databases
- **Gene Information**: Detailed gene annotations, descriptions, and metadata
- **Sequence Data**: Access to gene sequences, transcripts, and protein data  
- **Reference Genomes**: Species-specific genomic reference information
- **Search Capabilities**: Find genes by symbols, keywords, or biological functions

## Documentation

- **[Complete Building Guide](docs/BUILDING_GUIDE.md)**: Comprehensive step-by-step documentation of the entire development process, including design decisions, safety framework implementation, local model setup, and beginner-friendly tutorials
- **[Local Model Setup](docs/BUILDING_GUIDE.md#local-model-setup)**: Instructions for cost-effective deployment with Qwen2.5-Coder-3B
- **[AI Safety Framework](docs/BUILDING_GUIDE.md#ai-safety-framework)**: Detailed explanation of safety controls and boundaries

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) for local AI processing

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/georgiesamaha/bio-nerd-tool.git
cd bio-nerd-tool

# Install Python dependencies
pip install -e .

# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Download the AI model (llama3:8b - ~4.7GB)
ollama pull llama3:8b

# Set up the bio-nerd command
chmod +x ~/.local/bin/bio-nerd
```

### Verify Installation

```bash
# Test the AI model
ollama run llama3:8b "What is the TP53 gene?"

# Test bio-nerd (without AI first)
bio-nerd query "TP53"

# Test with AI interpretation
bio-nerd query "tell me about BRCA1" --ai
```

## Usage

### Basic Queries

```bash
# Simple gene lookup
bio-nerd query "TP53"
bio-nerd query "ENSG00000141510"

# Natural language queries with AI interpretation
bio-nerd query "tell me about the BRCA1 gene" --ai
bio-nerd query "what does TP53 do?" --ai
bio-nerd query "EGFR function and mutations" --ai

# Multiple genes
bio-nerd query "compare TP53 and BRCA1" --ai
```

### Advanced Features

```bash
# Disable AI interpretation for faster queries
bio-nerd query "BRCA2" --no-ai

# Interactive mode
bio-nerd
# Then type queries interactively

# Get help
bio-nerd --help
bio-nerd query --help
```

### Example Output

```
🔬 Processing: tell me about TP53
--------------------------------------------------
🎯 Gene IDs detected: TP53
🔗 Ensembl ID mappings:
   TP53 → ENSG00000141510

📊 Gene Information:
• TP53:
   ✅ tumor protein p53
   📝 tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]

🧠 AI Interpretation:
TP53 is one of the most important tumor suppressor genes in human biology, often called the "guardian of the genome." Located on chromosome 17, it encodes the p53 protein which acts as a transcription factor that regulates the cell cycle and prevents cancer formation...
```

## Core Capabilities

### Gene Information (`gget.info`)
- Comprehensive gene annotations from Ensembl  
- Gene descriptions, biotypes, and synonyms
- Chromosome locations and genomic coordinates
- Protein coding information and domains

### Gene Search (`gget.search`) 
- Find genes by symbols, names, or keywords
- Species-specific searches (default: human)
- Fuzzy matching for partial gene names
- Alias and synonym resolution

### Sequence Data (`gget.seq`)
- DNA, RNA, and protein sequences
- Transcript isoform sequences  
- UTR and coding sequence regions
- FASTA format output

### Reference Genomes (`gget.ref`)
- Download genome assemblies and annotations
- GTF/GFF3 annotation files
- Species-specific reference data
- Assembly metadata and statistics

### AI-Powered Features
- **Natural Language Processing**: Extract gene identifiers from conversational queries
- **Biological Context**: Explain gene functions, pathways, and disease associations  
- **Intelligent Filtering**: Distinguish real gene names from typos and common words
- **Multi-gene Analysis**: Compare and analyze multiple genes simultaneously

## Performance & Architecture 

### Speed Considerations
- **Database Access**: gget uses Ensembl REST API (10-15 seconds per query)
- **AI Processing**: Local llama3:8b inference (~1-3 seconds)
- **Optimisation**: Future versions may include local database caching for faster access

### Technical Architecture
- **FastMCP Framework**: Robust server infrastructure 
- **Async Processing**: Non-blocking query handling
- **Error Recovery**: Graceful handling of network timeouts and API failures
- **Memory Efficient**: Streaming responses for large datasets

### Gene Detection Pipeline
1. **Pattern Matching**: Regex detection of Ensembl IDs and gene symbols
2. **AI Validation**: llama3:8b confirms ambiguous candidates are real genes
3. **Database Resolution**: Map gene symbols to canonical Ensembl identifiers
4. **Data Retrieval**: Fetch comprehensive information via gget library

## Reliability & Data Quality

### Data Sources
- **Ensembl**: Primary source for gene annotations and genomic data
- **NCBI**: Complementary database for additional gene information  
- **Real-time Access**: Always fetches latest database versions
- **No Data Caching**: Ensures information is current (though slower)

### Quality Assurance 
- **Source Attribution**: All data includes original database references
- **Error Handling**: Clear reporting when information is unavailable
- **Gene Validation**: AI prevents misidentification of non-gene terms as genes
- **Precise Matching**: Exact gene symbol to Ensembl ID resolution

### Known Limitations
- **Network Dependency**: Requires internet connection for database access
- **API Rate Limits**: Ensembl REST API may throttle heavy usage
- **Query Speed**: 10-15 seconds typical for comprehensive gene information
- **Human-Focused**: Primarily optimized for human genome queries

## Development & Contributing

### Future Enhancements
- **Local Database**: PyEnsembl integration for faster queries (~100x speedup)
- **Multi-species Support**: Expanded beyond human genome
- **Batch Processing**: Handle multiple genes in single queries  
- **Visualization**: Integration with plotting libraries for gene data
- **API Extensions**: Additional gget tools (phylogenetic trees, mutations, etc.)

### Technical Roadmap
- **Performance**: Local database caching to eliminate API delays
- **Features**: Pathway analysis and gene network visualization
- **Integrations**: Connect with other bioinformatics tools and workflows
- **User Experience**: Enhanced error messages and query suggestions

## Project Structure

```
bio-nerd-tool/
├── gget_mcp/
│   ├── __init__.py
│   ├── server_simple.py    # Main FastMCP server with AI integration
│   ├── server.py           # Alternative MCP server implementation  
│   ├── safety/             # Safety framework components
│   │   ├── boundaries.py   # Domain boundaries and validation
│   │   ├── epistemic.py    # Confidence and uncertainty handling
│   │   └── failures.py     # Error management and recovery
│   ├── tools/              # Bioinformatics tool implementations
│   │   ├── gget_info.py    # Gene information queries
│   │   └── nl_gene_query.py # Natural language processing
│   ├── schemas/            # Data validation schemas  
│   └── nlp/               # Natural language processing components
│       └── query_processor.py
├── config/                 # Configuration files for various MCP clients
│   ├── claude_desktop_config.json
│   ├── lm_studio_config.json  
│   └── ollama_config.py
├── tests/                  # Test suite
├── docs/                   # Documentation
├── ~/.local/bin/bio-nerd  # CLI command wrapper
├── pyproject.toml         # Python package configuration
└── README.md
```

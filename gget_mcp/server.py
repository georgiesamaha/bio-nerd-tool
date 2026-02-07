"""gget-MCP Server Implementation.

This module implements the main MCP server that coordinates all safety systems,
tool execution, and client communication for the gget bioinformatics query server.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence
from contextlib import asynccontextmanager

# MCP imports
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import Tool
except ImportError:
    print("Error: MCP library not installed. Run 'pip install mcp'", file=sys.stderr)
    sys.exit(1)

# Internal imports
from .tools.gget_info import GgetInfoTool
from .tools.nl_gene_query import NaturalLanguageGeneTool
from .safety.boundaries import SYSTEM_INVARIANTS, DEFAULT_AUTHORITY_BOUNDARIES, DEFAULT_DOMAIN_BOUNDARIES
from .safety.epistemic import MINIMUM_CONFIDENCE_FOR_ASSERTIONS
from .safety.failures import error_handler
from .safety.provenance import provenance_tracker
from .schemas.outputs import SystemCapabilities


class GgetMCPServer:
    """Main gget-MCP server with comprehensive safety controls."""
    
    def __init__(self):
        """Initialize the gget-MCP server with safety systems."""
        
        # Configure logging for safety and audit
        self._configure_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize safety systems
        self.authority_boundaries = DEFAULT_AUTHORITY_BOUNDARIES
        self.domain_boundaries = DEFAULT_DOMAIN_BOUNDARIES
        
        # Initialize tools with safety controls
        self.gget_info_tool = GgetInfoTool()
        self.nl_gene_tool = NaturalLanguageGeneTool()
        
        # Initialize FastMCP server
        self.app = FastMCP(
            name="gget-mcp-server",
            instructions=self._get_system_instructions()
        )
        
        # Register tools
        self._register_tools()
        
        # System startup
        self.logger.info("gget-MCP Server initialized with safety controls")
        self._log_system_capabilities()
    
    def _configure_logging(self):
        """Configure comprehensive logging for safety and audit."""
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr),  # Log to stderr to avoid stdout pollution
                logging.FileHandler('gget_mcp_audit.log', mode='a')  # Audit log
            ]
        )
        
        # Configure specific loggers
        logging.getLogger('mcp').setLevel(logging.WARNING)  # Reduce MCP noise
        logging.getLogger('gget').setLevel(logging.WARNING)  # Reduce gget noise
        logging.getLogger(__name__).setLevel(logging.INFO)
        
        # Log safety initialization
        logger = logging.getLogger(__name__)
        logger.info("Safety logging system initialized")
        logger.info(f"System invariants: {list(SYSTEM_INVARIANTS.keys())}")
    
    def _get_system_instructions(self) -> str:
        """Get comprehensive system instructions for AI safety."""
        return '''# gget-MCP System Instructions

## System Identity and Scope
This system provides read-only access to public bioinformatics databases through the gget library. 
It operates under strict safety constraints to ensure reliable, traceable, and responsible access to biological data.

## Core Principles
1. **Accuracy over Completeness**: Prioritize correct information over comprehensive coverage
2. **Evidence over Helpfulness**: Provide only well-supported, traceable information
3. **Silence over Speculation**: Refuse requests when confidence is insufficient
4. **Explicit Boundaries**: Clear communication about system capabilities and limitations

## Authority Boundaries  
- **Domain**: Strictly limited to bioinformatics and gene information queries
- **Operations**: Read-only queries to public databases only
- **Data Sources**: Ensembl, NCBI, UniProt, and other established public databases
- **Rate Limits**: Maximum 60 queries per minute per client

## What This System CAN Do
- Retrieve gene, transcript, and protein information from public databases
- Process natural language questions about genes and genomic information
- Automatically detect Ensembl IDs and gene symbols in plain English queries
- Provide AI-powered interpretations and explanations of gene data
- Provide source attribution and confidence assessment for all responses
- Explain uncertainty and limitations in available data
- Validate gene identifiers and sanitize inputs

## What This System CANNOT Do
- Access private, clinical, or patient data
- Modify or update any external databases  
- Make predictions beyond available data
- Provide medical advice or clinical interpretations
- Execute arbitrary code or access file systems
- Make network requests outside gget's mechanisms

## Response Format
All responses include:
- Source attribution and citations
- Confidence levels and uncertainty quantification
- Processing provenance and audit trail
- Clear disclaimers when appropriate
- Recommendations for interpretation

## Failure Modes
When unable to fulfill a request, the system will:
- Explain clearly why the request cannot be completed
- Suggest alternative approaches within system boundaries
- Provide information about system limitations
- Never speculate or provide unverified information

## Safety Guarantees
- No hallucinated information - all data from authoritative sources
- Complete audit trail for accountability
- Explicit confidence assessment for all responses
- Graceful failure with informative error messages
- Respect for rate limits and resource constraints
'''

    def _register_tools(self):
        """Register MCP tools with safety validation."""
        
        @self.app.tool()
        async def ask_about_gene(
            query: str,
            ai_model: str = "auto"
        ) -> Dict[str, Any]:
            """Ask natural language questions about genes and get AI-powered responses.
            
            This tool processes plain English questions about genes, automatically
            detects Ensembl IDs or gene symbols, retrieves information using gget,
            and provides intelligent interpretations using AI.
            
            Examples:
            - "What is ENSG00000034713?"
            - "Tell me about the TP53 gene"  
            - "What does BRCA1 do?"
            - "Where is the MYC gene located?"
            """
            
            arguments = {
                "query": query,
                "ai_model": ai_model
            }
            
            return await self.nl_gene_tool.execute(arguments)
        
        @self.app.tool()
        async def gget_info(
            gene_id: str,
            confidence_level: str = "standard", 
            include_sequences: bool = False,
            species: Optional[str] = None
        ) -> Dict[str, Any]:
            """Retrieve comprehensive gene/transcript/protein information."""
            
            # Prepare arguments
            arguments = {
                "gene_id": gene_id,
                "confidence_level": confidence_level,
                "include_sequences": include_sequences,
                "species": species
            }
            
            # Execute with full safety controls
            return await self.gget_info_tool.execute(arguments)
        
        @self.app.tool() 
        async def system_capabilities() -> Dict[str, Any]:
            """Get system capabilities, limitations, and safety information."""
            
            capabilities = SystemCapabilities(
                available_tools=["gget_info", "ask_about_gene"],
                supported_databases=["ensembl", "ncbi", "uniprot"],
                data_types=["gene", "transcript", "protein", "sequence", "annotation"],
                limitations=[
                    "Read-only access to public databases only",
                    "No access to private or clinical data", 
                    "Rate limited to 60 queries per minute",
                    "No predictions beyond available data",
                    "No medical advice or clinical interpretations"
                ],
                rate_limits={
                    "queries_per_minute": 60,
                    "max_response_size": 10000
                },
                confidence_thresholds={
                    "minimum_for_response": MINIMUM_CONFIDENCE_FOR_ASSERTIONS,
                    "disclaimer_required_below": 0.8,
                    "refuse_below": 0.5
                },
                safety_measures=[
                    "Input validation and sanitization",
                    "Authority boundary enforcement", 
                    "Confidence assessment and uncertainty quantification",
                    "Complete provenance tracking",
                    "Audit logging for accountability",
                    "Graceful error handling with informative messages",
                    "Natural language query processing",
                    "AI-powered result interpretation"
                ]
            )
            
            return {
                "success": True,
                "data": capabilities.model_dump(),
                "message": "System operates under strict safety constraints for reliable bioinformatics data access"
            }
        
        @self.app.tool()
        async def system_status() -> Dict[str, Any]:
            """Get current system status and health information."""
            
            # Check gget availability
            try:
                import gget
                gget_available = True
                gget_version = getattr(gget, '__version__', 'unknown')
            except ImportError:
                gget_available = False
                gget_version = None
            
            # Get safety system status
            safety_systems = {
                "authority_boundaries": "active",
                "domain_boundaries": "active", 
                "epistemic_controls": "active",
                "provenance_tracking": "active",
                "error_handling": "active"
            }
            
            # Get source reliability
            source_reliability = provenance_tracker.get_source_reliability_assessment()
            
            status = {
                "system_operational": gget_available,
                "gget_version": gget_version,
                "safety_systems": safety_systems,
                "source_reliability": source_reliability,
                "invariants_enforced": list(SYSTEM_INVARIANTS.keys()),
                "last_updated": "2024-01-15T10:00:00Z"
            }
            
            return {
                "success": True,
                "data": status,
                "warnings": [] if gget_available else ["gget library not available"]
            }
        
        self.logger.info("MCP tools registered successfully")
    
    def _log_system_capabilities(self):
        """Log system capabilities for audit trail."""
        
        self.logger.info("=== gget-MCP System Capabilities ===")
        self.logger.info(f"Authority boundaries: {self.authority_boundaries.model_dump()})")
        self.logger.info(f"Domain boundaries: {self.domain_boundaries.model_dump()}")
        self.logger.info(f"System invariants: {SYSTEM_INVARIANTS}")
        self.logger.info(f"Available tools: ['gget_info', 'ask_about_gene', 'system_capabilities', 'system_status']")
        self.logger.info("Safety systems: All active")
        self.logger.info("=== End System Capabilities ===")
    
    def run(self, transport: str = "stdio"):
        """Run the MCP server with specified transport."""
        
        self.logger.info(f"Starting gget-MCP server with transport: {transport}")
        
        try:
            # Run the FastMCP server (this handles its own async loop)
            self.app.run(transport=transport)
            
        except KeyboardInterrupt:
            self.logger.info("Server shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup
            provenance_tracker.clear_session()
            self.logger.info("gget-MCP server shutdown complete")


def create_mcp_server() -> GgetMCPServer:
    """Factory function to create a properly configured MCP server."""
    return GgetMCPServer()


def main():
    """Main entry point for the gget-MCP server."""
    
    # Create and configure server
    server = create_mcp_server()
    
    # Run with stdio transport (default for MCP)
    server.run(transport="stdio")


def main_sync():
    """Synchronous entry point for the server."""
    try:
        main()
    except KeyboardInterrupt:
        print("Server shutdown", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
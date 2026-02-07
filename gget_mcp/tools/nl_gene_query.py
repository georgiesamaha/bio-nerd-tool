"""Natural language gene query tool for gget-MCP.

This tool combines NLP processing with AI interpretation to handle
plain English questions about genes and provide intelligent responses.
"""

import logging
import time
from typing import Any, Dict, Optional
from uuid import uuid4

# Internal imports
from ..nlp.query_processor import GeneQueryProcessor, GeneQueryIntent
from ..ai import OllamaAgent, MockAIAgent, create_ai_agent, AIResponse
from .gget_info import GgetInfoTool
from ..schemas.outputs import create_safe_success_response, create_safe_error_response
from ..safety.boundaries import validate_invariants, OperationType, DomainScope
from ..safety.epistemic import assess_gene_info_confidence
from ..safety.failures import error_handler, RefusalReason
from ..safety.provenance import provenance_tracker


class NaturalLanguageGeneTool:
    """MCP tool for handling natural language gene queries."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.query_processor = GeneQueryProcessor()
        self.gget_tool = GgetInfoTool()
        self.ai_agent = None  # Will be initialized on first use
        
    async def _ensure_ai_agent(self):
        """Ensure AI agent is initialized."""
        if self.ai_agent is None:
            self.ai_agent = await create_ai_agent()
            agent_type = type(self.ai_agent).__name__
            self.logger.info(f"Initialized AI agent: {agent_type}")
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute natural language gene query with AI interpretation.
        
        Args:
            arguments: Contains 'query' (natural language question) and optional parameters
            
        Returns:
            Safe response with AI-interpreted gene information
        """
        request_id = str(uuid4())
        start_time = time.time()
        
        try:
            # Extract and validate input
            query = arguments.get("query", "").strip()
            if not query:
                return create_safe_error_response(
                    "Empty query provided",
                    request_id=request_id,
                    refusal_reason=RefusalReason.INVALID_INPUT
                )
            
            self.logger.info(f"Processing natural language query: {query[:100]}...")
            
            # Step 1: Process the natural language query
            intent = self.query_processor.process_query(query)
            
            # Step 2: Check if this is a valid gene query
            if intent.confidence < 0.3:
                return self._create_non_gene_query_response(query, intent, request_id)
            
            # Step 3: Handle different query types
            if intent.suggested_action == "run_gget_info" and intent.ensembl_ids:
                return await self._handle_ensembl_query(query, intent, request_id, start_time)
            elif intent.suggested_action == "search_gene_symbol" and intent.gene_symbols:
                return await self._handle_gene_symbol_query(query, intent, request_id, start_time)
            else:
                return self._create_clarification_response(query, intent, request_id)
                
        except Exception as e:
            self.logger.error(f"Natural language query processing failed: {str(e)}", exc_info=True)
            return create_safe_error_response(
                "Failed to process natural language query",
                request_id=request_id,
                error_details=str(e)
            )
    
    async def _handle_ensembl_query(self, query: str, intent: GeneQueryIntent, request_id: str, start_time: float) -> Dict[str, Any]:
        """Handle queries with explicit Ensembl IDs."""
        
        # Use the first Ensembl ID found
        gene_id = intent.ensembl_ids[0]
        self.logger.info(f"Processing Ensembl ID: {gene_id}")
        
        # Step 1: Get gene information using gget
        gget_args = {
            "gene_id": gene_id,
            "confidence_level": "standard",
            "include_sequences": False,
            "species": None
        }
        
        gget_response = await self.gget_tool.execute(gget_args)
        
        if not gget_response.get("success"):
            return gget_response  # Return the error from gget_tool
        
        # Step 2: Initialize AI agent and get interpretation
        await self._ensure_ai_agent()
        
        # Extract the actual gene data from gget response
        gene_data = gget_response.get("data", {}).get("gene_information", {})
        
        # Step 3: Get AI interpretation
        if intent.intent_type == "gene_info":
            ai_response = await self.ai_agent.interpret_gene_info(gene_id, gene_data)
        else:
            ai_response = await self.ai_agent.answer_gene_question(query, gene_id, gene_data)
        
        # Step 4: Create comprehensive response
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response_data = {
            "natural_language_response": ai_response.content,
            "query_analysis": {
                "original_query": intent.original_query,
                "detected_intent": intent.intent_type,
                "confidence": intent.confidence,
                "ensembl_ids_found": intent.ensembl_ids,
                "gene_symbols_found": intent.gene_symbols
            },
            "gene_data": {
                "source": "gget_info",
                "gene_id": gene_id,
                "raw_data": gene_data
            },
            "ai_analysis": {
                "model_used": ai_response.model_used,
                "ai_confidence": ai_response.confidence,
                "processing_time_ms": ai_response.processing_time_ms,
                "success": ai_response.success
            },
            "processing_metrics": {
                "total_processing_time_ms": processing_time_ms,
                "nlp_confidence": intent.confidence
            }
        }
        
        # Create provenance record
        provenance_record = provenance_tracker.track_gene_info_query(
            gene_id, 
            {"gget_data": gene_data, "ai_interpretation": ai_response.content}
        )
        
        return create_safe_success_response(
            data=response_data,
            epistemic_state=gget_response.get("epistemic_state"),
            provenance_record=provenance_record,
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )
    
    async def _handle_gene_symbol_query(self, query: str, intent: GeneQueryIntent, request_id: str, start_time: float) -> Dict[str, Any]:
        """Handle queries with gene symbols (requires symbol-to-ID lookup)."""
        
        # For now, suggest the user provide an Ensembl ID
        # This could be enhanced with a gene symbol lookup service
        
        gene_symbols = ", ".join(intent.gene_symbols[:3])  # Show first 3 symbols
        
        response_content = f"""I detected you're asking about gene symbol(s): **{gene_symbols}**

To provide the most accurate information, I need the Ensembl ID for this gene. Here's how to find it:

**Option 1: Use Ensembl website**
1. Go to https://www.ensembl.org/
2. Search for "{intent.gene_symbols[0]}"  
3. Find the Ensembl Gene ID (starts with ENSG)
4. Ask me again with: "What is ENSG00000XXXXX?"

**Option 2: Use gget search** 
```bash
gget search "{intent.gene_symbols[0]}"
```

**Your original question**: {query}

Once you have the Ensembl ID, I can provide detailed information and analysis about the gene!
"""
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response_data = {
            "natural_language_response": response_content,
            "query_analysis": {
                "original_query": query,
                "detected_intent": intent.intent_type,
                "confidence": intent.confidence,
                "gene_symbols_found": intent.gene_symbols,
                "suggestion": "provide_ensembl_id"
            },
            "processing_metrics": {
                "total_processing_time_ms": processing_time_ms,
                "nlp_confidence": intent.confidence
            }
        }
        
        return create_safe_success_response(
            data=response_data,
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )
    
    def _create_non_gene_query_response(self, query: str, intent: GeneQueryIntent, request_id: str) -> Dict[str, Any]:
        """Create response for queries that don't appear to be about genes."""
        
        response_content = f"""I'm a specialized bioinformatics assistant focused on gene information from public databases.

**Your query**: {query}

This doesn't appear to be a question about genes or genomic information (confidence: {intent.confidence:.2f}).

**I can help you with**:
- Gene information: "What is ENSG00000034713?"
- Gene function: "What does the TP53 gene do?"  
- Gene details: "Tell me about BRCA1"
- Gene location: "Where is ENSG00000141510 located?"

**To ask about a specific gene**:
1. Use an Ensembl ID (ENSG00000XXXXX)
2. Use a common gene symbol (TP53, BRCA1, etc.)
3. Ask in plain English about gene function or characteristics

Would you like to ask a gene-related question instead?
"""
        
        response_data = {
            "natural_language_response": response_content,
            "query_analysis": {
                "original_query": query,
                "detected_intent": intent.intent_type,
                "confidence": intent.confidence,
                "suggestion": "ask_gene_question"
            }
        }
        
        return create_safe_success_response(
            data=response_data,
            request_id=request_id
        )
    
    def _create_clarification_response(self, query: str, intent: GeneQueryIntent, request_id: str) -> Dict[str, Any]:
        """Create response when query needs clarification."""
        
        response_content = f"""I understand you're asking about genes, but I need more specific information.

**Your query**: {query}
**Detected intent**: {intent.intent_type}
**Confidence**: {intent.confidence:.2f}

**To help you better, please provide**:
- A specific Ensembl ID (like ENSG00000034713)
- A gene symbol (like TP53 or BRCA1)  
- A more specific question about what you want to know

**Examples of clear questions**:
- "What is ENSG00000034713?"
- "What does the TP53 gene do?"
- "Tell me about BRCA1 function"
- "Where is the MYC gene located?"

What specific gene would you like to learn about?
"""
        
        response_data = {
            "natural_language_response": response_content,
            "query_analysis": {
                "original_query": query,
                "detected_intent": intent.intent_type,
                "confidence": intent.confidence,
                "suggestion": "provide_specific_gene"
            }
        }
        
        return create_safe_success_response(
            data=response_data,
            request_id=request_id
        )
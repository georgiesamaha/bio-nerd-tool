"""gget.info tool implementation with safety controls.

This module implements the gget.info functionality as an MCP tool with
comprehensive safety boundaries, error handling, and provenance tracking.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Third-party imports
try:
    import gget
except ImportError:
    gget = None

# Internal imports
from ..schemas.inputs import GgetInfoInput, validate_and_sanitize_input, check_input_safety
from ..schemas.outputs import (
    GgetInfoOutput, GgetInfoResponseData, SafeResponse, 
    create_safe_success_response, create_safe_error_response
)
from ..safety.boundaries import (
    DEFAULT_AUTHORITY_BOUNDARIES, DEFAULT_DOMAIN_BOUNDARIES,
    validate_invariants, OperationType, DomainScope
)
from ..safety.epistemic import (
    assess_gene_info_confidence, create_uncertainty_aware_response,
    should_refuse_low_confidence_query
)
from ..safety.failures import (
    error_handler, handle_invalid_gene_id, handle_data_not_found,
    handle_boundary_violation, RefusalReason
)
from ..safety.provenance import provenance_tracker


class GgetInfoTool:
    """MCP tool for gget.info functionality with comprehensive safety controls."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.authority_boundaries = DEFAULT_AUTHORITY_BOUNDARIES
        self.domain_boundaries = DEFAULT_DOMAIN_BOUNDARIES
        
        # Validate gget availability
        if gget is None:
            raise ImportError("gget library is required but not installed")
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gget info query with full safety controls.
        
        Args:
            arguments: Raw arguments from MCP client
            
        Returns:
            Safe response with full provenance and epistemic metadata
        """
        request_id = str(uuid4())
        start_time = time.time()
        
        try:
            # Step 1: Input validation and sanitization
            validated_input = self._validate_input(arguments)
            
            # Step 2: Safety and boundary checks
            safety_result = self._check_safety_boundaries(validated_input)
            if not safety_result["allowed"]:
                return self._create_refusal_response(safety_result["reason"], request_id)
            
            # Step 3: Execute gget query with error handling
            gget_response = await self._execute_gget_query(validated_input)
            
            # Step 4: Assess confidence and create epistemic state
            epistemic_state = assess_gene_info_confidence(
                gget_response, 
                source_database="ensembl"
            )
            
            # Step 5: Check if confidence meets requirements
            if should_refuse_low_confidence_query(epistemic_state.uncertainty.confidence_score):
                return self._create_low_confidence_refusal(validated_input.gene_id, request_id)
            
            # Step 6: Create provenance record
            provenance_record = provenance_tracker.track_gene_info_query(
                validated_input.gene_id, 
                gget_response
            )
            
            # Step 7: Structure response data
            response_data = self._structure_response_data(validated_input.gene_id, gget_response)
            
            # Step 8: Create safe response with full metadata
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return create_safe_success_response(
                data=response_data,
                epistemic_state=epistemic_state,
                provenance_record=provenance_record,
                processing_time_ms=processing_time_ms,
                request_id=request_id
            ).dict()
            
        except Exception as e:
            return self._handle_execution_error(e, arguments, request_id)
    
    def _validate_input(self, arguments: Dict[str, Any]) -> GgetInfoInput:
        """Validate and sanitize input arguments."""
        try:
            return validate_and_sanitize_input(arguments)
        except ValueError as e:
            error_response = error_handler.handle_failure(
                "invalid_input",
                context={"user_input": arguments, "validation_error": str(e)}
            )
            raise ValueError(f"Input validation failed: {str(e)}") from e
    
    def _check_safety_boundaries(self, input_obj: GgetInfoInput) -> Dict[str, Any]:
        """Check safety boundaries and authority constraints."""
        
        # Check authority boundaries
        if not self.authority_boundaries.is_operation_allowed(
            OperationType.READ_ONLY_QUERY, 
            DomainScope.BIOINFORMATICS
        ):
            return {
                "allowed": False,
                "reason": RefusalReason.PROHIBITED_ACTION,
                "message": "Read-only queries not authorized in bioinformatics domain"
            }
        
        # Check domain boundaries
        if not self.domain_boundaries.is_query_allowed(
            database="ensembl",
            data_type="gene",
            species=input_obj.species
        ):
            return {
                "allowed": False,
                "reason": RefusalReason.OUT_OF_SCOPE,
                "message": self.domain_boundaries.get_boundary_violation_message(
                    database="ensembl", data_type="gene", species=input_obj.species
                )
            }
        
        # Check system invariants
        invariant_results = validate_invariants(
            "query",
            gene_id=input_obj.gene_id,
            data_type="gene",
            allow_sequences=input_obj.include_sequences
        )
        
        violations = [k for k, v in invariant_results.items() if not v]
        if violations:
            return {
                "allowed": False,
                "reason": RefusalReason.SAFETY_VIOLATION,
                "message": f"Operation violates system invariants: {', '.join(violations)}"
            }
        
        # Check input safety
        safety_checks = check_input_safety(input_obj)
        safety_violations = [k for k, v in safety_checks.items() if not v]
        if safety_violations:
            return {
                "allowed": False,
                "reason": RefusalReason.INVALID_INPUT,
                "message": f"Input fails safety checks: {', '.join(safety_violations)}"
            }
        
        return {"allowed": True}
    
    async def _execute_gget_query(self, input_obj: GgetInfoInput) -> Dict[str, Any]:
        """Execute the actual gget query with error handling."""
        try:
            self.logger.info(f"Executing gget.info for gene_id: {input_obj.gene_id}")
            
            # Execute gget.info with appropriate parameters
            result = gget.info(
                input_obj.gene_id,
                verbose=False  # Reduce noise in output
            )
            
            # gget.info returns a pandas DataFrame, convert to dict
            if hasattr(result, 'to_dict'):
                # Convert DataFrame to dict, taking first row if multiple
                result_dict = result.to_dict('records')
                if result_dict:
                    return result_dict[0]
                else:
                    return {}
            
            # If result is already a dict or other format
            return result if isinstance(result, dict) else {}
            
        except Exception as e:
            self.logger.error(f"gget.info query failed for {input_obj.gene_id}: {str(e)}")
            
            # Determine appropriate error type
            error_msg = str(e).lower()
            if "not found" in error_msg or "no results" in error_msg:
                raise ValueError(f"No data found for gene ID: {input_obj.gene_id}")
            elif "invalid" in error_msg or "format" in error_msg:
                raise ValueError(f"Invalid gene ID format: {input_obj.gene_id}")
            else:
                raise RuntimeError(f"gget query failed: {str(e)}")
    
    def _structure_response_data(self, gene_id: str, raw_data: Dict[str, Any]) -> GgetInfoResponseData:
        """Structure raw gget data into validated response format."""
        
        # Calculate completeness score
        expected_fields = ['gene_name', 'description', 'biotype', 'chromosome', 'start', 'end']
        present_fields = sum(1 for field in expected_fields if field in raw_data and raw_data.get(field))
        completeness_score = present_fields / len(expected_fields)
        
        # Extract GO terms if present
        go_terms = []
        if 'go_terms' in raw_data and raw_data['go_terms']:
            # Handle different GO term formats
            go_data = raw_data['go_terms']
            if isinstance(go_data, list):
                go_terms = [{
                    "id": term.get('id', ''),
                    "name": term.get('name', ''),
                    "namespace": term.get('namespace', '')
                } for term in go_data if isinstance(term, dict)]
        
        # Extract pathways if present
        pathways = []
        if 'pathways' in raw_data and raw_data['pathways']:
            pathway_data = raw_data['pathways']
            if isinstance(pathway_data, list):
                pathways = [{
                    "id": pathway.get('id', ''),
                    "name": pathway.get('name', ''),
                    "source": pathway.get('source', '')
                } for pathway in pathway_data if isinstance(pathway, dict)]
        
        return GgetInfoResponseData(
            gene_id=gene_id,
            gene_name=raw_data.get('gene_name'),
            description=raw_data.get('description'),
            biotype=raw_data.get('biotype'),
            chromosome=str(raw_data.get('chromosome')) if raw_data.get('chromosome') is not None else None,
            start_position=int(raw_data.get('start')) if raw_data.get('start') is not None else None,
            end_position=int(raw_data.get('end')) if raw_data.get('end') is not None else None,
            strand=str(raw_data.get('strand')) if raw_data.get('strand') is not None else None,
            ensembl_id=raw_data.get('ensembl_gene_id'),
            ncbi_id=str(raw_data.get('ncbi_gene_id')) if raw_data.get('ncbi_gene_id') is not None else None,
            uniprot_ids=raw_data.get('uniprot_ids') if isinstance(raw_data.get('uniprot_ids'), list) else None,
            synonyms=raw_data.get('synonyms') if isinstance(raw_data.get('synonyms'), list) else None,
            go_terms=go_terms if go_terms else None,
            pathways=pathways if pathways else None,
            completeness_score=completeness_score
        )
    
    def _create_refusal_response(self, reason: RefusalReason, request_id: str) -> Dict[str, Any]:
        """Create a standardized refusal response."""
        error_response = error_handler.create_refusal(reason)
        return create_safe_error_response(error_response, request_id).dict()
    
    def _create_low_confidence_refusal(self, gene_id: str, request_id: str) -> Dict[str, Any]:
        """Create refusal for low confidence data."""
        error_response = error_handler.create_refusal(
            RefusalReason.INSUFFICIENT_CONFIDENCE,
            context={"user_input": {"gene_id": gene_id}},
            custom_message=f"Available data for gene '{gene_id}' has insufficient confidence for reliable response"
        )
        return create_safe_error_response(error_response, request_id).dict()
    
    def _handle_execution_error(self, exception: Exception, arguments: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Handle execution errors with appropriate error responses."""
        
        error_msg = str(exception)
        
        # Determine error type and create appropriate response
        if "not found" in error_msg.lower() or "no data" in error_msg.lower():
            error_response = handle_data_not_found(
                gene_id=arguments.get('gene_id', 'unknown'),
                request_id=request_id
            )
        elif "invalid" in error_msg.lower() or "validation" in error_msg.lower():
            error_response = handle_invalid_gene_id(
                gene_id=arguments.get('gene_id', 'unknown'),
                request_id=request_id
            )
        elif "boundary" in error_msg.lower() or "prohibited" in error_msg.lower():
            error_response = handle_boundary_violation(
                operation="gget_info_query", 
                request_id=request_id
            )
        else:
            # Generic system error
            error_response = error_handler.handle_failure(
                "system_error",
                context={"user_input": arguments, "exception": str(exception)},
                exception=exception,
                request_id=request_id
            )
        
        return create_safe_error_response(error_response, request_id).dict()
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the MCP tool schema for this tool."""
        return {
            "name": "gget_info",
            "description": (
                "Retrieve comprehensive gene/transcript/protein information from public databases. "
                "This tool provides read-only access to curated bioinformatics data with full "
                "provenance tracking and confidence assessment. All responses include source "
                "attribution and uncertainty quantification to ensure responsible use of "
                "biological information."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gene_id": {
                        "type": "string",
                        "description": "Gene identifier (Ensembl ID, NCBI Gene ID, UniProt ID, or gene symbol)",
                        "examples": ["ENSG00000157764", "7157", "TP53", "P53_HUMAN"]
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["low", "standard", "high"],
                        "description": "Minimum confidence level required for response",
                        "default": "standard"
                    },
                    "include_sequences": {
                        "type": "boolean",
                        "description": "Whether to include sequence information (may reduce confidence)",
                        "default": False
                    },
                    "species": {
                        "type": "string",
                        "description": "Species filter (scientific name, e.g. 'homo sapiens')",
                        "examples": ["homo sapiens", "mus musculus", "drosophila melanogaster"]
                    }
                },
                "required": ["gene_id"],
                "additionalProperties": False
            }
        }
"""Output validation schemas for gget-MCP responses.

This module defines comprehensive output schemas that include safety metadata,
provenance information, and epistemic state for all responses.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from ..safety.epistemic import EpistemicState, ConfidenceLevel
from ..safety.provenance import ProvenanceRecord
from ..safety.failures import SafeErrorResponse


class ProvenanceMetadata(BaseModel):
    """Provenance metadata for response traceability."""
    
    sources: List[Dict[str, Any]] = Field(
        description="Source attributions for this response"
    )
    
    data_lineage_id: str = Field(
        description="Unique identifier for data lineage tracking"
    )
    
    processing_steps: List[str] = Field(
        description="Summary of processing steps applied"
    )
    
    last_updated: datetime = Field(
        description="When the underlying data was last updated"
    )
    
    citation_text: Optional[str] = Field(
        default=None,
        description="Formatted citation text for this response"
    )


class EpistemicMetadata(BaseModel):
    """Epistemic metadata for confidence and uncertainty."""
    
    confidence_level: ConfidenceLevel = Field(
        description="Overall confidence level in this response"
    )
    
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Numerical confidence score"
    )
    
    uncertainty_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to uncertainty"
    )
    
    evidence_quality: str = Field(
        description="Assessment of evidence quality"
    )
    
    requires_disclaimer: bool = Field(
        description="Whether this response requires a disclaimer"
    )
    
    disclaimer_text: Optional[str] = Field(
        default=None,
        description="Disclaimer text if required"
    )


class GgetInfoResponseData(BaseModel):
    """Core gene information response data."""
    
    gene_id: str = Field(
        description="Queried gene identifier"
    )
    
    gene_name: Optional[str] = Field(
        default=None,
        description="Official gene name/symbol"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Gene description"
    )
    
    biotype: Optional[str] = Field(
        default=None,
        description="Gene biotype (protein_coding, lncRNA, etc.)"
    )
    
    chromosome: Optional[str] = Field(
        default=None,
        description="Chromosome location"
    )
    
    start_position: Optional[int] = Field(
        default=None,
        description="Start position on chromosome"
    )
    
    end_position: Optional[int] = Field(
        default=None,
        description="End position on chromosome"
    )
    
    strand: Optional[str] = Field(
        default=None,
        description="Strand orientation (+/-)"
    )
    
    ensembl_id: Optional[str] = Field(
        default=None,
        description="Ensembl gene identifier"
    )
    
    ncbi_id: Optional[str] = Field(
        default=None,
        description="NCBI gene identifier"
    )
    
    uniprot_ids: Optional[List[str]] = Field(
        default=None,
        description="Associated UniProt protein identifiers"
    )
    
    synonyms: Optional[List[str]] = Field(
        default=None,
        description="Alternative gene names"
    )
    
    go_terms: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Gene Ontology terms"
    )
    
    pathways: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Associated biological pathways"
    )
    
    expression_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expression data if available"
    )
    
    completeness_score: float = Field(
        ge=0.0, le=1.0,
        description="Completeness score for available data"
    )


class GgetInfoOutput(BaseModel):
    """Complete output schema for gget info queries."""
    
    success: bool = Field(
        description="Whether the query was successful"
    )
    
    data: Optional[GgetInfoResponseData] = Field(
        default=None,
        description="Gene information data (null if query failed)"
    )
    
    epistemic_metadata: EpistemicMetadata = Field(
        description="Confidence and uncertainty information"
    )
    
    provenance_metadata: ProvenanceMetadata = Field(
        description="Source attribution and lineage information"
    )
    
    processing_time_ms: int = Field(
        description="Processing time in milliseconds"
    )
    
    timestamp: datetime = Field(
        description="Response generation timestamp"
    )
    
    system_version: str = Field(
        default="gget-mcp-0.1.0",
        description="System version that generated this response"
    )
    
    safety_flags: List[str] = Field(
        default_factory=list,
        description="Any safety flags or warnings for this response"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for interpreting or using this information"
    )


class SafeResponse(BaseModel):
    """Universal safe response wrapper for all operations."""
    
    success: bool = Field(
        description="Whether the operation was successful"
    )
    
    data: Optional[Any] = Field(
        default=None,
        description="Response data (null if operation failed)"
    )
    
    error: Optional[SafeErrorResponse] = Field(
        default=None,
        description="Error details if operation failed"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings about this operation"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this response"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracing"
    )


class SystemCapabilities(BaseModel):
    """System capabilities and limitations disclosure."""
    
    available_tools: List[str] = Field(
        description="List of available MCP tools"
    )
    
    supported_databases: List[str] = Field(
        description="Supported bioinformatics databases"
    )
    
    data_types: List[str] = Field(
        description="Types of biological data that can be queried"
    )
    
    limitations: List[str] = Field(
        description="System limitations and boundaries"
    )
    
    rate_limits: Dict[str, int] = Field(
        description="Rate limiting information"
    )
    
    confidence_thresholds: Dict[str, float] = Field(
        description="Confidence thresholds for different operations"
    )
    
    safety_measures: List[str] = Field(
        description="Description of implemented safety measures"
    )


def create_safe_success_response(
    data: Any,
    epistemic_state: EpistemicState,
    provenance_record: ProvenanceRecord,
    processing_time_ms: int,
    request_id: Optional[str] = None
) -> SafeResponse:
    """Create a safe success response with full metadata."""
    
    # Extract epistemic metadata
    epistemic_metadata = EpistemicMetadata(
        confidence_level=epistemic_state.confidence_level,
        confidence_score=epistemic_state.uncertainty.confidence_score,
        uncertainty_factors=[
            uncertainty_type.value for uncertainty_type in epistemic_state.uncertainty.uncertainty_types
        ],
        evidence_quality=epistemic_state.uncertainty.evidence_quality.value,
        requires_disclaimer=epistemic_state.requires_disclaimer,
        disclaimer_text=epistemic_state.disclaimer_text
    )
    
    # Extract provenance metadata
    provenance_metadata = ProvenanceMetadata(
        sources=[
            {
                "name": source.source_name,
                "type": source.source_type.value,
                "url": str(source.source_url) if source.source_url else None,
                "reliability_score": source.reliability_score,
                "citation": source.citation
            }
            for lineage in provenance_record.data_lineages
            for source in lineage.primary_sources
        ],
        data_lineage_id=provenance_record.data_lineages[0].data_id if provenance_record.data_lineages else "unknown",
        processing_steps=[
            step.description for lineage in provenance_record.data_lineages
            for step in lineage.processing_pipeline
        ],
        last_updated=provenance_record.response_timestamp
    )
    
    # Build recommendations
    recommendations = []
    if epistemic_state.confidence_level == ConfidenceLevel.LOW:
        recommendations.append("Verify this information with additional sources")
    if epistemic_state.uncertainty.conflicting_sources:
        recommendations.append("Multiple sources provide conflicting information")
    
    return SafeResponse(
        success=True,
        data=data,
        metadata={
            "epistemic": epistemic_metadata.dict(),
            "provenance": provenance_metadata.dict(),
            "processing_time_ms": processing_time_ms,
            "recommendations": recommendations
        },
        request_id=request_id
    )


def create_safe_error_response(
    error_response: SafeErrorResponse,
    request_id: Optional[str] = None
) -> SafeResponse:
    """Create a safe error response."""
    
    return SafeResponse(
        success=False,
        error=error_response,
        warnings=error_response.failure_mode.recovery_suggestions,
        metadata={
            "error_category": error_response.failure_mode.category.value,
            "error_severity": error_response.failure_mode.severity.value,
            "safe_to_retry": error_response.safe_to_retry
        },
        request_id=request_id
    )
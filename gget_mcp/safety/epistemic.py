"""Epistemic stance and confidence management for AI safety.

This module implements confidence levels, uncertainty quantification, and
epistemic state management to prevent overconfident assertions and clearly
communicate the reliability of information.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ConfidenceLevel(str, Enum):
    """Confidence levels for information reliability."""
    HIGH = "high"           # >95% confidence, multiple reliable sources
    STANDARD = "standard"   # 80-95% confidence, established sources
    LOW = "low"            # 50-80% confidence, limited sources  
    UNCERTAIN = "uncertain" # <50% confidence, conflicting or sparse sources
    UNKNOWN = "unknown"     # No reliable information available


class EvidenceQuality(str, Enum):
    """Quality assessment of supporting evidence."""
    PEER_REVIEWED = "peer_reviewed"
    DATABASE_VERIFIED = "database_verified" 
    COMPUTATIONAL_PREDICTION = "computational_prediction"
    EXPERIMENTAL_VALIDATION = "experimental_validation"
    LITERATURE_REFERENCE = "literature_reference"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class UncertaintyType(str, Enum):
    """Types of uncertainty in biological data."""
    MEASUREMENT_ERROR = "measurement_error"
    INCOMPLETE_DATA = "incomplete_data"
    CONFLICTING_SOURCES = "conflicting_sources"
    OUTDATED_INFORMATION = "outdated_information"
    ANNOTATION_QUALITY = "annotation_quality"
    SPECIES_VARIATION = "species_variation"
    PREDICTION_UNCERTAINTY = "prediction_uncertainty"


class UncertaintyQuantification(BaseModel):
    """Quantifies and explains uncertainty in responses."""
    
    uncertainty_types: List[UncertaintyType] = Field(
        default_factory=list,
        description="Types of uncertainty present in the data"
    )
    
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Numerical confidence score (0.0 = no confidence, 1.0 = complete confidence)"
    )
    
    evidence_quality: EvidenceQuality = Field(
        description="Quality assessment of supporting evidence"
    )
    
    source_count: int = Field(
        ge=0,
        description="Number of independent sources supporting this information"
    )
    
    last_updated: Optional[datetime] = Field(
        default=None,
        description="When the underlying data was last updated"
    )
    
    conflicting_sources: List[str] = Field(
        default_factory=list,
        description="List of sources that provide conflicting information"
    )
    
    gaps_in_knowledge: List[str] = Field(
        default_factory=list,
        description="Identified gaps or limitations in available knowledge"
    )
    
    @validator('confidence_score')
    def validate_confidence_consistency(cls, v, values):
        """Ensure confidence score aligns with evidence quality."""
        if 'evidence_quality' in values:
            quality = values['evidence_quality']
            if quality == EvidenceQuality.PEER_REVIEWED and v < 0.8:
                raise ValueError("Peer-reviewed evidence should have confidence ≥ 0.8")
            elif quality == EvidenceQuality.INFERRED and v > 0.6:
                raise ValueError("Inferred evidence should have confidence ≤ 0.6")
        return v


class EpistemicState(BaseModel):
    """Complete epistemic state for a piece of information."""
    
    confidence_level: ConfidenceLevel = Field(
        description="Overall confidence level in this information"
    )
    
    uncertainty: UncertaintyQuantification = Field(
        description="Detailed uncertainty quantification"
    )
    
    should_speculate: bool = Field(
        default=False,
        description="Whether speculation beyond available data is appropriate"
    )
    
    requires_disclaimer: bool = Field(
        default=True, 
        description="Whether this information requires a disclaimer"
    )
    
    disclaimer_text: Optional[str] = Field(
        default=None,
        description="Specific disclaimer text if required"
    )
    
    @validator('disclaimer_text')
    def generate_disclaimer_if_needed(cls, v, values):
        """Generate appropriate disclaimer text if needed."""
        if values.get('requires_disclaimer', True) and v is None:
            confidence = values.get('confidence_level')
            
            if confidence == ConfidenceLevel.UNCERTAIN:
                return "This information has low confidence and may be incomplete or inaccurate."
            elif confidence == ConfidenceLevel.LOW:
                return "This information has limited supporting evidence and should be verified."
            elif confidence == ConfidenceLevel.UNKNOWN:
                return "No reliable information is available for this query."
            else:
                return "This information is based on available database sources and may be subject to updates."
                
        return v


class EpistemicResponse(BaseModel):
    """Wrapper for responses with epistemic metadata."""
    
    data: Any = Field(
        description="The actual response data"
    )
    
    epistemic_state: EpistemicState = Field(
        description="Epistemic metadata about the response"
    )
    
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning that led to this response"
    )
    
    alternative_interpretations: List[str] = Field(
        default_factory=list,
        description="Alternative ways to interpret the available data"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for interpreting or using this information"
    )


def assess_gene_info_confidence(
    gene_data: Dict[str, Any],
    source_database: str = "ensembl"
) -> EpistemicState:
    """Assess confidence for gene information queries.
    
    Args:
        gene_data: Raw gene data from gget
        source_database: Primary source database used
        
    Returns:
        EpistemicState with appropriate confidence assessment
    """
    # Initialize uncertainty quantification
    uncertainties = []
    confidence_score = 0.9  # Start high for database data
    evidence_quality = EvidenceQuality.DATABASE_VERIFIED
    source_count = 1
    
    # Assess data completeness
    if not gene_data or len(gene_data) == 0:
        return EpistemicState(
            confidence_level=ConfidenceLevel.UNKNOWN,
            uncertainty=UncertaintyQuantification(
                uncertainty_types=[UncertaintyType.INCOMPLETE_DATA],
                confidence_score=0.0,
                evidence_quality=EvidenceQuality.UNKNOWN,
                source_count=0,
                gaps_in_knowledge=["No data available for this gene identifier"]
            )
        )
    
    # Check for missing critical fields
    critical_fields = ['gene_name', 'description', 'biotype']
    missing_fields = [field for field in critical_fields if field not in gene_data or not gene_data[field]]
    
    if missing_fields:
        uncertainties.append(UncertaintyType.INCOMPLETE_DATA)
        confidence_score *= 0.85
    
    # Check data recency (if available)
    if 'version' in gene_data or 'last_updated' in gene_data:
        # Data with version info is more reliable
        evidence_quality = EvidenceQuality.DATABASE_VERIFIED
        source_count += 1
    else:
        uncertainties.append(UncertaintyType.OUTDATED_INFORMATION)
        confidence_score *= 0.9
    
    # Assess annotation quality
    if gene_data.get('biotype') in ['protein_coding', 'lncRNA', 'miRNA']:
        # Well-characterized biotypes
        pass
    elif gene_data.get('biotype') in ['pseudogene', 'processed_pseudogene']:
        uncertainties.append(UncertaintyType.ANNOTATION_QUALITY)
        confidence_score *= 0.8
    else:
        uncertainties.append(UncertaintyType.ANNOTATION_QUALITY)
        confidence_score *= 0.7
    
    # Determine confidence level
    if confidence_score >= 0.95:
        confidence_level = ConfidenceLevel.HIGH
    elif confidence_score >= 0.8:
        confidence_level = ConfidenceLevel.STANDARD
    elif confidence_score >= 0.5:
        confidence_level = ConfidenceLevel.LOW
    else:
        confidence_level = ConfidenceLevel.UNCERTAIN
    
    # Build uncertainty quantification
    uncertainty = UncertaintyQuantification(
        uncertainty_types=uncertainties,
        confidence_score=confidence_score,
        evidence_quality=evidence_quality,
        source_count=source_count,
        gaps_in_knowledge=[f"Missing {field}" for field in missing_fields]
    )
    
    return EpistemicState(
        confidence_level=confidence_level,
        uncertainty=uncertainty,
        should_speculate=False,  # Never speculate beyond available data
        requires_disclaimer=confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN, ConfidenceLevel.UNKNOWN]
    )


def create_uncertainty_aware_response(
    data: Any,
    confidence_assessment: EpistemicState,
    reasoning_steps: List[str] = None
) -> EpistemicResponse:
    """Create a response with full epistemic awareness.
    
    Args:
        data: The response data
        confidence_assessment: Epistemic state assessment
        reasoning_steps: Optional reasoning trace
        
    Returns:
        EpistemicResponse with appropriate metadata
    """
    recommendations = []
    
    # Generate context-appropriate recommendations
    if confidence_assessment.confidence_level == ConfidenceLevel.LOW:
        recommendations.append("Verify this information with additional sources")
        recommendations.append("Consider the identified uncertainties when interpreting results")
    
    if confidence_assessment.confidence_level == ConfidenceLevel.UNCERTAIN:
        recommendations.append("This information should not be used for critical decisions")
        recommendations.append("Consult domain experts for interpretation")
    
    if confidence_assessment.uncertainty.conflicting_sources:
        recommendations.append("Multiple sources provide different information - exercise caution")
    
    return EpistemicResponse(
        data=data,
        epistemic_state=confidence_assessment,
        reasoning_trace=reasoning_steps or [],
        recommendations=recommendations
    )


# Epistemic safety constants
MINIMUM_CONFIDENCE_FOR_ASSERTIONS = 0.7
SPECULATION_PROHIBITED_THRESHOLD = 0.3  # Lowered from 0.5 for more permissive testing
DISCLAIMER_REQUIRED_THRESHOLD = 0.8

def should_refuse_low_confidence_query(confidence_score: float) -> bool:
    """Determine if a query should be refused due to low confidence."""
    return confidence_score < SPECULATION_PROHIBITED_THRESHOLD
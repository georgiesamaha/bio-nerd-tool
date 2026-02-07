"""Input validation schemas for gget-MCP tools.

This module defines strict input validation to ensure all requests
are well-formed and within system boundaries.
"""

import re
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from ..safety.epistemic import ConfidenceLevel


class GgetInfoInput(BaseModel):
    """Input schema for gget info queries.
    
    Validates gene identifiers and confidence requirements.
    """
    
    gene_id: str = Field(
        description="Gene identifier (Ensembl, NCBI, UniProt, or gene symbol)",
        min_length=1,
        max_length=50,
        example="ENSG00000157764"
    )
    
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.STANDARD,
        description="Minimum confidence level required for response"
    )
    
    include_sequences: bool = Field(
        default=False,
        description="Whether to include sequence information (may reduce confidence)"
    )
    
    species: Optional[str] = Field(
        default=None,
        description="Species filter for multi-species gene families",
        max_length=100
    )
    
    @validator('gene_id')
    def validate_gene_id_format(cls, v):
        """Validate gene identifier format and safety."""
        if not v or not v.strip():
            raise ValueError("Gene ID cannot be empty")
        
        # Remove potentially dangerous characters
        if any(char in v for char in ['<', '>', '&', '"', "'", ';']):
            raise ValueError("Gene ID contains invalid characters")
        
        # Normalize whitespace
        v = v.strip()
        
        # Validate against known patterns
        if not validate_gene_identifier(v):
            raise ValueError(
                f"Gene ID '{v}' doesn't match recognized formats. "
                "Expected: Ensembl (ENSG...), NCBI (numeric), UniProt, or gene symbol"
            )
        
        return v
    
    @validator('species')
    def validate_species_format(cls, v):
        """Validate species name format if provided."""
        if v is None:
            return v
        
        v = v.strip()
        
        # Check for dangerous characters
        if any(char in v for char in ['<', '>', '&', '"', "'", ';']):
            raise ValueError("Species name contains invalid characters")
        
        # Basic format validation (scientific name pattern)
        if not re.match(r'^[A-Za-z][a-z]+(\s+[a-z]+)*$', v):
            raise ValueError(
                "Species should be in scientific name format (e.g., 'homo sapiens')"
            )
        
        return v.lower()  # Normalize to lowercase
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        str_strip_whitespace = True
        anystr_lower = False  # Keep gene IDs case-sensitive


def validate_gene_identifier(gene_id: str) -> bool:
    """Validate gene identifier against known formats.
    
    Args:
        gene_id: Gene identifier to validate
        
    Returns:
        True if the identifier matches a recognized format
    """
    if not gene_id or not isinstance(gene_id, str):
        return False
    
    # Ensembl gene ID (e.g., ENSG00000157764)
    ensembl_pattern = r'^ENS[A-Z]*G[0-9]{11}$'
    if re.match(ensembl_pattern, gene_id):
        return True
    
    # Ensembl transcript ID (e.g., ENST00000288602)
    ensembl_transcript_pattern = r'^ENS[A-Z]*T[0-9]{11}$'
    if re.match(ensembl_transcript_pattern, gene_id):
        return True
    
    # Ensembl protein ID (e.g., ENSP00000288602)
    ensembl_protein_pattern = r'^ENS[A-Z]*P[0-9]{11}$'
    if re.match(ensembl_protein_pattern, gene_id):
        return True
    
    # NCBI Gene ID (numeric)
    ncbi_pattern = r'^[0-9]+$'
    if re.match(ncbi_pattern, gene_id) and len(gene_id) <= 12:
        return True
    
    # UniProt ID (e.g., P04637, A0A0B4J2F2)
    uniprot_pattern = r'^[A-NR-Z][0-9]([A-Z][A-Z, 0-9][A-Z, 0-9][0-9]){1,2}$|^[OPQ][0-9][A-Z, 0-9][A-Z, 0-9][A-Z, 0-9][0-9](\.\d+)?$'
    if re.match(uniprot_pattern, gene_id):
        return True
    
    # RefSeq accession (e.g., NM_000546, NP_000537)
    refseq_pattern = r'^(NM_|NR_|XM_|XR_|NP_|XP_|WP_|YP_|AP_|NZ_)[0-9]+(\.\d+)?$'
    if re.match(refseq_pattern, gene_id):
        return True
    
    # Gene symbol (basic validation - letters, numbers, hyphens, underscores)
    # More permissive but still safe
    symbol_pattern = r'^[A-Za-z][A-Za-z0-9_-]{1,20}$'
    if re.match(symbol_pattern, gene_id):
        return True
    
    return False


# Input validation utilities
class InputValidationError(ValueError):
    """Custom exception for input validation errors."""
    pass


def validate_and_sanitize_input(input_data: dict) -> GgetInfoInput:
    """Validate and sanitize input data.
    
    Args:
        input_data: Raw input dictionary
        
    Returns:
        Validated and sanitized input object
        
    Raises:
        InputValidationError: If validation fails
    """
    try:
        return GgetInfoInput(**input_data)
    except ValueError as e:
        raise InputValidationError(f"Input validation failed: {str(e)}")


def check_input_safety(input_obj: GgetInfoInput) -> dict:
    """Perform additional safety checks on validated input.
    
    Args:
        input_obj: Validated input object
        
    Returns:
        Dictionary of safety check results
    """
    safety_checks = {
        "gene_id_safe": True,
        "species_safe": True,
        "confidence_appropriate": True,
        "within_rate_limits": True  # This would be checked elsewhere
    }
    
    # Check gene ID for additional safety concerns
    gene_id = input_obj.gene_id
    if len(gene_id) > 30 or any(ord(c) > 127 for c in gene_id):
        safety_checks["gene_id_safe"] = False
    
    # Check species safety if provided
    if input_obj.species:
        species = input_obj.species
        if len(species) > 50 or any(ord(c) > 127 for c in species):
            safety_checks["species_safe"] = False
    
    # Check confidence level appropriateness
    if input_obj.include_sequences and input_obj.confidence_level == ConfidenceLevel.HIGH:
        safety_checks["confidence_appropriate"] = False
    
    return safety_checks
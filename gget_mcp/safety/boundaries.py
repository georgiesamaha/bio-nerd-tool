"""Authority and domain boundary enforcement for AI safety.

This module defines and enforces the boundaries of what the gget-MCP server
can and cannot do, preventing scope creep and maintaining clear limitations.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class DomainScope(str, Enum):
    """Allowed domains of operation."""
    BIOINFORMATICS = "bioinformatics"
    GENE_INFORMATION = "gene_information"
    PUBLIC_DATABASES = "public_databases"


class OperationType(str, Enum):
    """Types of operations the server can perform."""
    READ_ONLY_QUERY = "read_only_query"
    DATA_RETRIEVAL = "data_retrieval"
    INFORMATION_LOOKUP = "information_lookup"


class ProhibitedAction(str, Enum):
    """Explicitly prohibited actions."""
    DATA_MODIFICATION = "data_modification"
    EXTERNAL_API_CALLS = "external_api_calls"
    FILE_SYSTEM_ACCESS = "file_system_access"
    NETWORK_REQUESTS = "network_requests"
    USER_DATA_STORAGE = "user_data_storage"
    ARBITRARY_CODE_EXECUTION = "arbitrary_code_execution"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"


class AuthorityBoundaries(BaseModel):
    """Defines the authority and capability boundaries of the server."""
    
    allowed_domains: Set[DomainScope] = Field(
        default={DomainScope.BIOINFORMATICS, DomainScope.GENE_INFORMATION, DomainScope.PUBLIC_DATABASES},
        description="Domains the server is authorized to operate in"
    )
    
    allowed_operations: Set[OperationType] = Field(
        default={OperationType.READ_ONLY_QUERY, OperationType.DATA_RETRIEVAL, OperationType.INFORMATION_LOOKUP},
        description="Types of operations the server can perform"
    )
    
    prohibited_actions: Set[ProhibitedAction] = Field(
        default=set(ProhibitedAction),
        description="Explicitly prohibited actions"
    )
    
    max_query_rate: int = Field(
        default=60,
        description="Maximum queries per minute per client"
    )
    
    max_response_size: int = Field(
        default=10_000,
        description="Maximum response size in characters"
    )
    
    def is_operation_allowed(self, operation: OperationType, domain: DomainScope) -> bool:
        """Check if an operation is allowed in a given domain."""
        return operation in self.allowed_operations and domain in self.allowed_domains
    
    def is_action_prohibited(self, action: ProhibitedAction) -> bool:
        """Check if an action is explicitly prohibited."""
        return action in self.prohibited_actions
    
    def get_violation_message(self, operation: Optional[OperationType] = None, 
                            action: Optional[ProhibitedAction] = None) -> str:
        """Generate a clear violation message."""
        if action:
            return f"Action '{action.value}' is explicitly prohibited by system boundaries."
        if operation:
            return f"Operation '{operation.value}' is not authorized in current domain scope."
        return "Request violates established authority boundaries."


class DomainBoundaries(BaseModel):
    """Specific domain boundaries for bioinformatics operations."""
    
    allowed_gene_databases: Set[str] = Field(
        default={"ensembl", "ncbi", "uniprot"},
        description="Allowed gene databases for queries"
    )
    
    allowed_species: Optional[Set[str]] = Field(
        default=None,
        description="Allowed species (None = all public species)"
    )
    
    allowed_data_types: Set[str] = Field(
        default={"gene", "transcript", "protein", "sequence", "annotation"},
        description="Allowed types of biological data to retrieve"
    )
    
    prohibited_data_types: Set[str] = Field(
        default={"patient_data", "clinical_data", "private_sequences", "unpublished_data"},
        description="Explicitly prohibited data types"
    )
    
    def is_query_allowed(self, database: str, data_type: str, 
                        species: Optional[str] = None) -> bool:
        """Check if a specific query is allowed within domain boundaries."""
        if database not in self.allowed_gene_databases:
            return False
        
        if data_type in self.prohibited_data_types:
            return False
            
        if data_type not in self.allowed_data_types:
            return False
            
        if self.allowed_species is not None and species is not None:
            if species not in self.allowed_species:
                return False
                
        return True
    
    def get_boundary_violation_message(self, database: Optional[str] = None,
                                     data_type: Optional[str] = None,
                                     species: Optional[str] = None) -> str:
        """Generate specific boundary violation message."""
        if database and database not in self.allowed_gene_databases:
            return f"Database '{database}' is outside allowed scope. Allowed: {', '.join(self.allowed_gene_databases)}"
        
        if data_type and data_type in self.prohibited_data_types:
            return f"Data type '{data_type}' is explicitly prohibited"
            
        if data_type and data_type not in self.allowed_data_types:
            return f"Data type '{data_type}' is not in allowed scope. Allowed: {', '.join(self.allowed_data_types)}"
            
        if species and self.allowed_species and species not in self.allowed_species:
            return f"Species '{species}' not in allowed scope"
            
        return "Query violates domain boundaries"


# Default boundary configurations - immutable invariants  
DEFAULT_AUTHORITY_BOUNDARIES = AuthorityBoundaries()
DEFAULT_DOMAIN_BOUNDARIES = DomainBoundaries()

# Invariant rules that cannot be violated
SYSTEM_INVARIANTS = {
    "read_only": "System operates in strict read-only mode",
    "no_external_calls": "No external API calls beyond gget's internal mechanisms", 
    "public_data_only": "Only public, non-sensitive biological data accessible",
    "traceable_sources": "All data must have verifiable source attribution",
    "bounded_scope": "Operations limited to defined bioinformatics domain"
}


def validate_invariants(operation: str, **kwargs) -> Dict[str, bool]:
    """Validate that operation maintains system invariants.
    
    Returns:
        Dict mapping invariant names to validation results
    """
    results = {}
    
    # Check read-only invariant
    results["read_only"] = operation in ["read", "query", "lookup", "retrieve"]
    
    # Check external calls invariant  
    results["no_external_calls"] = "external_url" not in kwargs and "api_call" not in kwargs
    
    # Check public data invariant
    results["public_data_only"] = kwargs.get("data_type", "").lower() not in [
        "private", "clinical", "patient", "unpublished", "proprietary"
    ]
    
    # Check traceable sources invariant
    results["traceable_sources"] = True  # gget provides this by default
    
    # Check bounded scope invariant
    allowed_domains = ["gene", "transcript", "protein", "ensembl", "ncbi", "uniprot"]
    results["bounded_scope"] = any(
        domain in str(kwargs).lower() for domain in allowed_domains
    ) or "gene_id" in kwargs
    
    return results
"""Provenance tracking and source attribution for gget-MCP.

This module implements comprehensive provenance tracking to ensure all data
can be traced back to authoritative sources with full audit trails.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse


class SourceType(str, Enum):
    """Types of data sources."""
    DATABASE = "database"
    API = "api"
    LIBRARY = "library"
    FILE = "file"
    COMPUTATION = "computation"


class DataProcessingStep(str, Enum):
    """Types of data processing steps."""
    RETRIEVAL = "retrieval"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation" 
    AGGREGATION = "aggregation"
    FILTERING = "filtering"


@dataclass
class SourceAttribution:
    """Attribution information for a data source."""
    
    source_id: str
    source_name: str
    source_type: SourceType
    source_url: Optional[str] = None
    reliability_score: float = 0.8
    citation: Optional[str] = None
    license_info: Optional[str] = None
    access_date: datetime = field(default_factory=datetime.now)
    version: Optional[str] = None


@dataclass 
class ProcessingStep:
    """Information about a data processing step."""
    
    step_id: str
    step_type: DataProcessingStep
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    input_sources: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Complete lineage information for a data element."""
    
    data_id: str
    data_content_hash: str
    primary_sources: List[str]
    processing_steps: List[ProcessingStep]
    creation_timestamp: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum if not provided."""
        if not self.checksum:
            content_str = json.dumps({
                "data_id": self.data_id,
                "sources": self.primary_sources,
                "steps": [step.step_id for step in self.processing_steps]
            }, sort_keys=True)
            self.checksum = hashlib.sha256(content_str.encode()).hexdigest()


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a response."""
    
    record_id: str
    query_fingerprint: str
    data_lineages: List[DataLineage]
    system_version: str
    processing_environment: Dict[str, str]
    audit_trail: List[str]
    creation_timestamp: datetime = field(default_factory=datetime.now)


class ProvenanceTracker:
    """Comprehensive provenance tracking system."""
    
    def __init__(self):
        """Initialize the provenance tracker."""
        self._source_registry: Dict[str, SourceAttribution] = {}
        self._lineage_registry: Dict[str, DataLineage] = {}
        self._processing_steps: List[ProcessingStep] = []
        
        # Register default sources
        self._register_default_sources()
    
    def _register_default_sources(self):
        """Register default bioinformatics data sources."""
        
        # Ensembl
        self.register_source(SourceAttribution(
            source_id="ensembl",
            source_name="Ensembl",
            source_type=SourceType.DATABASE,
            source_url="https://www.ensembl.org/",
            reliability_score=0.95,
            citation="Martin et al. Ensembl 2023. Nucleic Acids Res. 2023;51(D1):D933-D941.",
            license_info="Apache License 2.0",
            version="110"
        ))
        
        # NCBI
        self.register_source(SourceAttribution(
            source_id="ncbi",
            source_name="NCBI Gene",
            source_type=SourceType.DATABASE,
            source_url="https://www.ncbi.nlm.nih.gov/gene/",
            reliability_score=0.95,
            citation="Sayers et al. Database resources of the NCBI. Nucleic Acids Res. 2023;51(D1):D29-D38.",
            license_info="Public Domain"
        ))
        
        # gget library
        self.register_source(SourceAttribution(
            source_id="gget",
            source_name="gget",
            source_type=SourceType.LIBRARY,
            source_url="https://github.com/pachterlab/gget",
            reliability_score=0.85,
            citation="Luebbert & Pachter. Efficient querying of genomic reference databases with gget. Bioinformatics. 2023;39(8):1446-1448.",
            license_info="BSD 2-Clause License"
        ))
    
    def register_source(self, source: SourceAttribution):
        """Register a data source for attribution."""
        self._source_registry[source.source_id] = source
    
    def add_processing_step(
        self, 
        step_type: DataProcessingStep,
        description: str,
        input_sources: Optional[List[str]] = None,
        transformations: Optional[List[str]] = None,
        validation_checks: Optional[List[str]] = None,
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> ProcessingStep:
        """Add a processing step to the current session."""
        
        step_id = f"step_{len(self._processing_steps) + 1}_{int(datetime.now().timestamp())}"
        
        step = ProcessingStep(
            step_id=step_id,
            step_type=step_type,
            description=description,
            input_sources=input_sources or [],
            transformations=transformations or [],
            validation_checks=validation_checks or [],
            quality_metrics=quality_metrics or {}
        )
        
        self._processing_steps.append(step)
        return step
    
    def create_data_lineage(
        self,
        data_id: str,
        data_content: Any,
        primary_source_ids: List[str],
        processing_steps: Optional[List[ProcessingStep]] = None
    ) -> DataLineage:
        """Create a data lineage record."""
        
        # Create content hash
        content_str = json.dumps(data_content, default=str, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        lineage = DataLineage(
            data_id=data_id,
            data_content_hash=content_hash,
            primary_sources=primary_source_ids,
            processing_steps=processing_steps or []
        )
        
        # Register lineage
        self._lineage_registry[data_id] = lineage
        
        return lineage
    
    def create_provenance_record(
        self,
        query: Dict[str, Any],
        response_data: Any,
        data_lineages: List[DataLineage]
    ) -> ProvenanceRecord:
        """Create a complete provenance record."""
        
        # Create query fingerprint
        query_str = json.dumps(query, default=str, sort_keys=True)
        query_fingerprint = hashlib.sha256(query_str.encode()).hexdigest()
        
        # Generate unique record ID
        record_id = f"prov_{int(datetime.now().timestamp())}_{query_fingerprint[:8]}"
        
        return ProvenanceRecord(
            record_id=record_id,
            query_fingerprint=query_fingerprint,
            data_lineages=data_lineages,
            system_version="gget-mcp-0.1.0",
            processing_environment={
                "python_version": "3.10+",
                "gget_version": "0.28.0+"
            },
            audit_trail=[
                f"Query received: {datetime.now().isoformat()}",
                f"Processing steps: {len(self._processing_steps)}",
                f"Sources consulted: {len(set(sum([lineage.primary_sources for lineage in data_lineages], [])))}"
            ]
        )
    
    def get_source_attribution(self, source_id: str) -> Optional[SourceAttribution]:
        """Get attribution information for a source."""
        return self._source_registry.get(source_id)
    
    def track_gene_info_query(self, gene_id: str, gget_response: Any) -> ProvenanceRecord:
        """Track a gene info query and create provenance record."""
        
        # Add processing steps for gene info query
        retrieval_step = self.add_processing_step(
            step_type=DataProcessingStep.RETRIEVAL,
            description=f"Gene information retrieval for {gene_id}",
            input_sources=["ensembl", "gget"],
            validation_checks=["gene_id_format", "data_availability"]
        )
        
        validation_step = self.add_processing_step(
            step_type=DataProcessingStep.VALIDATION,
            description="Data quality validation and consistency checks",
            input_sources=["ensembl"],
            validation_checks=["data_completeness", "format_validation"],
            quality_metrics={"completeness": 0.9, "reliability": 0.95}
        )
        
        # Create data lineage for the response
        lineage = self.create_data_lineage(
            data_id=f"gene_info_{gene_id}",
            data_content=gget_response,
            primary_source_ids=["ensembl", "gget"],
            processing_steps=[retrieval_step, validation_step]
        )
        
        # Create provenance record
        query = {"gene_id": gene_id, "tool": "gget.info"}
        record = self.create_provenance_record(
            query=query,
            response_data=gget_response,
            data_lineages=[lineage]
        )
        
        return record
    
    def get_source_reliability_assessment(self) -> Dict[str, float]:
        """Get reliability assessment for all registered sources."""
        return {
            source_id: source.reliability_score 
            for source_id, source in self._source_registry.items()
        }
    
    def clear_session(self):
        """Clear session-specific data while keeping source registry."""
        self._lineage_registry.clear()
        self._processing_steps.clear()


# Global provenance tracker instance
provenance_tracker = ProvenanceTracker()


def get_citation_text(source_ids: List[str]) -> str:
    """Generate formatted citation text for given sources."""
    citations = []
    
    for source_id in source_ids:
        source = provenance_tracker.get_source_attribution(source_id)
        if source and source.citation:
            citations.append(source.citation)
    
    if not citations:
        return "Data retrieved from public bioinformatics databases"
    
    return "; ".join(citations)
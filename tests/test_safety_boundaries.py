"""Tests for safety boundary enforcement."""

import pytest
from gget_mcp.safety.boundaries import (
    AuthorityBoundaries, DomainBoundaries, validate_invariants,
    OperationType, DomainScope, ProhibitedAction
)


class TestAuthorityBoundaries:
    """Test authority boundary enforcement."""
    
    def test_default_boundaries_allow_bioinformatics(self):
        boundaries = AuthorityBoundaries()
        assert boundaries.is_operation_allowed(
            OperationType.READ_ONLY_QUERY, 
            DomainScope.BIOINFORMATICS
        )
    
    def test_prohibited_actions_rejected(self):
        boundaries = AuthorityBoundaries()
        assert boundaries.is_action_prohibited(ProhibitedAction.DATA_MODIFICATION)
        assert boundaries.is_action_prohibited(ProhibitedAction.ARBITRARY_CODE_EXECUTION)
    
    def test_violation_messages_informative(self):
        boundaries = AuthorityBoundaries()
        message = boundaries.get_violation_message(action=ProhibitedAction.DATA_MODIFICATION)
        assert "data_modification" in message.lower()
        assert "prohibited" in message.lower()


class TestDomainBoundaries:
    """Test domain-specific boundary enforcement."""
    
    def test_allowed_databases_accepted(self):
        boundaries = DomainBoundaries()
        assert boundaries.is_query_allowed("ensembl", "gene")
        assert boundaries.is_query_allowed("ncbi", "protein")
    
    def test_prohibited_data_types_rejected(self):
        boundaries = DomainBoundaries()
        assert not boundaries.is_query_allowed("ensembl", "patient_data")
        assert not boundaries.is_query_allowed("ncbi", "clinical_data")
    
    def test_unknown_databases_rejected(self):
        boundaries = DomainBoundaries()
        assert not boundaries.is_query_allowed("unknown_db", "gene")


class TestInvariantValidation:
    """Test system invariant validation."""
    
    def test_read_only_operations_pass(self):
        results = validate_invariants("read", gene_id="ENSG00000157764")
        assert results["read_only"] is True
        assert results["bounded_scope"] is True
    
    def test_modification_operations_fail(self):
        results = validate_invariants("write", gene_id="ENSG00000157764")
        assert results["read_only"] is False
    
    def test_external_calls_detected(self):
        results = validate_invariants("read", external_url="http://example.com")
        assert results["no_external_calls"] is False
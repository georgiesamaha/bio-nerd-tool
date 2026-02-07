"""Failure mode handling and refusal templates for AI safety.

This module implements comprehensive error handling, graceful failure modes,
and standardized refusal templates to ensure the system fails safely and
provides clear communication about its limitations.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
import logging
import traceback
from datetime import datetime


class FailureCategory(str, Enum):
    """Categories of system failures."""
    INPUT_VALIDATION = "input_validation"
    BOUNDARY_VIOLATION = "boundary_violation"  
    DATA_UNAVAILABLE = "data_unavailable"
    EXTERNAL_SERVICE = "external_service"
    RATE_LIMIT = "rate_limit"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_ERROR = "authentication_error"


class RefusalReason(str, Enum):
    """Standardized reasons for refusing requests."""
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    PROHIBITED_ACTION = "prohibited_action"
    INVALID_INPUT = "invalid_input"  
    RATE_LIMITED = "rate_limited"
    DATA_QUALITY_CONCERNS = "data_quality_concerns"
    SAFETY_VIOLATION = "safety_violation"
    TECHNICAL_LIMITATION = "technical_limitation"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class FailureMode(BaseModel):
    """Represents a specific failure mode with handling strategy."""
    
    category: FailureCategory = Field(
        description="Category of this failure mode"
    )
    
    severity: ErrorSeverity = Field(
        description="Severity level of this failure"
    )
    
    user_message: str = Field(
        description="User-facing error message"
    )
    
    technical_details: Optional[str] = Field(
        default=None,
        description="Technical details for debugging (not shown to user)"
    )
    
    recovery_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for how the user might recover"
    )
    
    should_retry: bool = Field(
        default=False,
        description="Whether this operation might succeed if retried"
    )
    
    retry_delay_seconds: Optional[int] = Field(
        default=None,
        description="Suggested delay before retry if should_retry is True"
    )


class RefusalTemplate(BaseModel):
    """Template for refusing requests with clear explanation."""
    
    reason: RefusalReason = Field(
        description="Standardized reason for refusal"
    )
    
    message: str = Field(
        description="Clear, non-apologetic explanation of why request was refused"
    )
    
    alternative_suggestions: List[str] = Field(
        default_factory=list,
        description="Alternative actions the user could take"
    )
    
    boundary_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of the specific boundary that prevents this action"
    )
    
    scope_clarification: Optional[str] = Field(
        default=None,
        description="Clarification of what the system can and cannot do"
    )


class ErrorContext(BaseModel):
    """Context information for errors and failures."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this request"
    )
    
    user_input: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized user input that caused the error"
    )
    
    system_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Relevant system state information"
    )
    
    stack_trace: Optional[str] = Field(
        default=None,
        description="Technical stack trace for debugging"
    )


class SafeErrorResponse(BaseModel):
    """Safe error response that doesn't leak internal details."""
    
    success: bool = Field(default=False)
    
    failure_mode: FailureMode = Field(
        description="Details about the failure mode"
    )
    
    refusal_template: Optional[RefusalTemplate] = Field(
        default=None,
        description="Refusal template if this was a refused request"
    )
    
    error_context: ErrorContext = Field(
        description="Context information for debugging"
    )
    
    safe_to_retry: bool = Field(
        description="Whether it's safe for the user to retry this request"
    )


class ErrorHandler:
    """Central error handling and failure management system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._failure_modes = self._initialize_failure_modes()
        self._refusal_templates = self._initialize_refusal_templates()
    
    def _initialize_failure_modes(self) -> Dict[str, FailureMode]:
        """Initialize standard failure modes."""
        return {
            "invalid_gene_id": FailureMode(
                category=FailureCategory.INPUT_VALIDATION,
                severity=ErrorSeverity.WARNING,
                user_message="The gene identifier provided is not valid or recognized",
                recovery_suggestions=[
                    "Check the gene ID format (e.g., ENSG00000157764 for Ensembl)",
                    "Verify the gene exists in public databases",
                    "Try using an alternative gene identifier format"
                ],
                should_retry=False
            ),
            
            "data_not_found": FailureMode(
                category=FailureCategory.DATA_UNAVAILABLE,
                severity=ErrorSeverity.INFO,
                user_message="No information available for the requested gene",
                recovery_suggestions=[
                    "Try a different gene identifier",
                    "Verify the gene exists in current database versions",
                    "Check if the gene has been deprecated or merged"
                ],
                should_retry=False
            ),
            
            "rate_limit_exceeded": FailureMode(
                category=FailureCategory.RATE_LIMIT,
                severity=ErrorSeverity.WARNING,
                user_message="Rate limit exceeded. Please wait before making additional requests",
                recovery_suggestions=[
                    "Wait 60 seconds before retrying",
                    "Reduce the frequency of requests"
                ],
                should_retry=True,
                retry_delay_seconds=60
            ),
            
            "boundary_violation": FailureMode(
                category=FailureCategory.BOUNDARY_VIOLATION,
                severity=ErrorSeverity.ERROR,
                user_message="Request violates system boundary constraints",
                recovery_suggestions=[
                    "Review the allowed operations for this system",
                    "Modify your request to stay within system boundaries"
                ],
                should_retry=False
            ),
            
            "service_unavailable": FailureMode(
                category=FailureCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.ERROR,
                user_message="External data service is temporarily unavailable",
                recovery_suggestions=[
                    "Try again in a few minutes",
                    "Check if the issue is widespread"
                ],
                should_retry=True,
                retry_delay_seconds=300
            ),
            
            "system_error": FailureMode(
                category=FailureCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.CRITICAL,
                user_message="An internal system error occurred",
                recovery_suggestions=[
                    "Report this error with the error ID",
                    "Try a different request"
                ],
                should_retry=False
            )
        }
    
    def _initialize_refusal_templates(self) -> Dict[RefusalReason, RefusalTemplate]:
        """Initialize standardized refusal templates."""
        return {
            RefusalReason.OUT_OF_SCOPE: RefusalTemplate(
                reason=RefusalReason.OUT_OF_SCOPE,
                message="This request is outside the system's defined scope of bioinformatics gene information queries.",
                alternative_suggestions=[
                    "Focus your query on gene, transcript, or protein information",
                    "Use recognized gene identifiers from public databases"
                ],
                scope_clarification="This system provides read-only access to public gene information from established bioinformatics databases."
            ),
            
            RefusalReason.INSUFFICIENT_CONFIDENCE: RefusalTemplate(
                reason=RefusalReason.INSUFFICIENT_CONFIDENCE,
                message="Available data has insufficient confidence for a reliable response.",
                alternative_suggestions=[
                    "Try a more specific gene identifier", 
                    "Check if newer data might be available in the source databases"
                ],
                boundary_explanation="The system prioritizes accuracy over completeness and will not provide low-confidence information."
            ),
            
            RefusalReason.PROHIBITED_ACTION: RefusalTemplate(
                reason=RefusalReason.PROHIBITED_ACTION,
                message="The requested action is explicitly prohibited by system safety constraints.",
                alternative_suggestions=[
                    "Review the system's allowed operations",
                    "Reformulate your request within allowed boundaries"
                ],
                boundary_explanation="The system operates under strict safety constraints that prevent certain types of operations."
            ),
            
            RefusalReason.INVALID_INPUT: RefusalTemplate(
                reason=RefusalReason.INVALID_INPUT,
                message="The input provided does not meet validation requirements.",
                alternative_suggestions=[
                    "Check the required input format",
                    "Ensure all required fields are provided",
                    "Verify input values are within acceptable ranges"
                ]
            ),
            
            RefusalReason.RATE_LIMITED: RefusalTemplate(
                reason=RefusalReason.RATE_LIMITED,
                message="Request frequency exceeds allowed limits.",
                alternative_suggestions=[
                    "Wait before making additional requests",
                    "Batch multiple queries if possible"
                ],
                boundary_explanation="Rate limiting ensures fair access and prevents system overload."
            ),
            
            RefusalReason.DATA_QUALITY_CONCERNS: RefusalTemplate(
                reason=RefusalReason.DATA_QUALITY_CONCERNS,
                message="Available data quality is insufficient to meet system reliability standards.",
                alternative_suggestions=[
                    "Try requesting different information",
                    "Check for alternative data sources"
                ],
                boundary_explanation="The system maintains strict data quality standards to prevent misinformation."
            ),
            
            RefusalReason.SAFETY_VIOLATION: RefusalTemplate(
                reason=RefusalReason.SAFETY_VIOLATION,
                message="Request violates AI safety constraints.",
                boundary_explanation="The system includes comprehensive safety measures that prevent certain types of requests."
            ),
            
            RefusalReason.TECHNICAL_LIMITATION: RefusalTemplate(
                reason=RefusalReason.TECHNICAL_LIMITATION,
                message="Current system capabilities cannot fulfill this request.",
                alternative_suggestions=[
                    "Try a simpler or more specific request",
                    "Break complex requests into smaller parts"
                ]
            )
        }
    
    def handle_failure(
        self, 
        failure_type: str, 
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        request_id: Optional[str] = None
    ) -> SafeErrorResponse:
        """Handle a failure and return a safe error response."""
        
        # Get failure mode or create generic one
        failure_mode = self._failure_modes.get(failure_type)
        if not failure_mode:
            failure_mode = FailureMode(
                category=FailureCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.ERROR,
                user_message="An unexpected error occurred",
                should_retry=False
            )
        
        # Create error context
        error_context = ErrorContext(
            request_id=request_id,
            user_input=context.get("user_input") if context else None,
            system_state=context.get("system_state") if context else None,
            stack_trace=traceback.format_exc() if exception else None
        )
        
        # Log the error appropriately
        self._log_error(failure_mode, error_context, exception)
        
        return SafeErrorResponse(
            failure_mode=failure_mode,
            error_context=error_context,
            safe_to_retry=failure_mode.should_retry
        )
    
    def create_refusal(
        self, 
        reason: RefusalReason,
        context: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> SafeErrorResponse:
        """Create a standardized refusal response."""
        
        template = self._refusal_templates[reason]
        
        # Override message if provided
        if custom_message:
            template = template.model_copy()
            template.message = custom_message
        
        # Create associated failure mode
        failure_mode = FailureMode(
            category=FailureCategory.BOUNDARY_VIOLATION,
            severity=ErrorSeverity.WARNING,
            user_message=template.message,
            recovery_suggestions=template.alternative_suggestions,
            should_retry=False
        )
        
        error_context = ErrorContext(
            user_input=context.get("user_input") if context else None
        )
        
        return SafeErrorResponse(
            failure_mode=failure_mode,
            refusal_template=template,
            error_context=error_context,
            safe_to_retry=False
        )
    
    def _log_error(
        self, 
        failure_mode: FailureMode, 
        context: ErrorContext,
        exception: Optional[Exception] = None
    ):
        """Log error with appropriate level and sanitization."""
        
        log_data = {
            "category": failure_mode.category,
            "severity": failure_mode.severity,
            "timestamp": context.timestamp.isoformat(),
            "request_id": context.request_id
        }
        
        # Log at appropriate level
        if failure_mode.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical failure: {failure_mode.user_message}", extra=log_data)
        elif failure_mode.severity == ErrorSeverity.ERROR:
            self.logger.error(f"Error: {failure_mode.user_message}", extra=log_data)
        elif failure_mode.severity == ErrorSeverity.WARNING:
            self.logger.warning(f"Warning: {failure_mode.user_message}", extra=log_data)
        else:
            self.logger.info(f"Info: {failure_mode.user_message}", extra=log_data)
        
        # Log exception details separately if present
        if exception:
            self.logger.debug(f"Exception details", exc_info=exception, extra=log_data)


# Global error handler instance
error_handler = ErrorHandler()


def handle_graceful_failure(failure_type: str, **kwargs) -> SafeErrorResponse:
    """Convenience function for graceful failure handling."""
    return error_handler.handle_failure(failure_type, kwargs)


def refuse_request(reason: RefusalReason, **kwargs) -> SafeErrorResponse:
    """Convenience function for standardized request refusal.""" 
    return error_handler.create_refusal(reason, kwargs)


# Pre-configured failure handlers for common scenarios
def handle_invalid_gene_id(gene_id: str, **context) -> SafeErrorResponse:
    """Handle invalid gene ID scenario."""
    return handle_graceful_failure(
        "invalid_gene_id",
        user_input={"gene_id": gene_id},
        **context
    )


def handle_data_not_found(gene_id: str, **context) -> SafeErrorResponse:
    """Handle data not found scenario."""
    return handle_graceful_failure(
        "data_not_found", 
        user_input={"gene_id": gene_id},
        **context
    )


def handle_boundary_violation(operation: str, **context) -> SafeErrorResponse:
    """Handle boundary violation scenario.""" 
    return refuse_request(
        RefusalReason.PROHIBITED_ACTION,
        user_input={"operation": operation},
        **context
    )
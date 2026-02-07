"""Natural Language Processing module for gget-MCP.

This module provides capabilities for understanding and processing
natural language queries about genes and biological information.
"""

from .query_processor import GeneQueryProcessor, GeneQueryIntent

__all__ = ['GeneQueryProcessor', 'GeneQueryIntent']
"""Natural Language Processing for gene queries.

This module handles parsing plain English questions about genes and
extracting actionable information for gget queries.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GeneQueryIntent:
    """Represents a parsed gene query intent."""
    
    intent_type: str  # "gene_info", "gene_function", "gene_location", etc.
    ensembl_ids: List[str]
    gene_symbols: List[str]
    confidence: float
    original_query: str
    suggested_action: str


class GeneQueryProcessor:
    """Processes natural language queries about genes."""
    
    def __init__(self):
        # Common Ensembl ID patterns
        self.ensembl_patterns = [
            r'\bENSG\d{11}\b',  # Ensembl Gene IDs
            r'\bENST\d{11}\b',  # Ensembl Transcript IDs  
            r'\bENSP\d{11}\b',  # Ensembl Protein IDs
        ]
        
        # Gene symbol patterns (basic)
        self.gene_symbol_pattern = r'\b[A-Z][A-Z0-9]{1,10}\b'
        
        # Intent keywords
        self.intent_keywords = {
            "gene_info": [
                "what is", "tell me about", "information about",
                "details about", "describe", "explain",
                "gene info", "gene information"
            ],
            "gene_function": [
                "function", "what does", "role", "purpose",
                "involved in", "responsible for"
            ],
            "gene_location": [
                "location", "chromosome", "position", "where",
                "located on", "genomic position"
            ],
            "gene_expression": [
                "expression", "expressed in", "tissue",
                "when expressed", "expression pattern"
            ]
        }
        
    def extract_ensembl_ids(self, text: str) -> List[str]:
        """Extract Ensembl IDs from text."""
        ensembl_ids = []
        
        for pattern in self.ensembl_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ensembl_ids.extend(matches)
            
        return list(set(ensembl_ids))  # Remove duplicates
    
    def extract_gene_symbols(self, text: str) -> List[str]:
        """Extract potential gene symbols from text."""
        # This is a basic implementation - could be enhanced with a gene symbol database
        potential_symbols = re.findall(self.gene_symbol_pattern, text)
        
        # Filter out common English words that match the pattern
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
            'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'WHAT',
            'USE', 'EACH', 'WHICH', 'SHE', 'DO', 'HOW', 'IF', 'UP',
            'SO', 'ABOUT', 'OUT', 'MANY', 'TIME', 'VERY', 'WHEN',
            'MUCH', 'NEW', 'NOW', 'OLD', 'SEE', 'HIM', 'TWO', 'WAY',
            'WHO', 'ITS', 'SAY', 'DID', 'GET', 'MAY', 'DAY'
        }
        
        filtered_symbols = [s for s in potential_symbols if s not in common_words]
        return filtered_symbols
    
    def detect_intent(self, text: str) -> str:
        """Detect the main intent of the query."""
        text_lower = text.lower()
        
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            intent_scores[intent] = score
            
        if not intent_scores or max(intent_scores.values()) == 0:
            return "gene_info"  # Default intent
            
        return max(intent_scores, key=intent_scores.get)
    
    def calculate_confidence(self, query: str, ensembl_ids: List[str], gene_symbols: List[str], intent: str) -> float:
        """Calculate confidence score for the parsed query."""
        confidence = 0.0
        
        # Boost for explicit Ensembl IDs
        if ensembl_ids:
            confidence += 0.8
            
        # Boost for potential gene symbols
        if gene_symbols:
            confidence += 0.4
            
        # Boost for clear intent keywords
        query_lower = query.lower()
        intent_keywords = self.intent_keywords.get(intent, [])
        for keyword in intent_keywords:
            if keyword in query_lower:
                confidence += 0.2
                break
                
        # Boost for gene-related terms
        gene_terms = ["gene", "protein", "transcript", "genomic", "dna", "sequence"]
        for term in gene_terms:
            if term in query_lower:
                confidence += 0.1
                break
                
        return min(confidence, 1.0)  # Cap at 1.0
    
    def process_query(self, query: str) -> GeneQueryIntent:
        """Process a natural language gene query."""
        
        # Extract identifiers
        ensembl_ids = self.extract_ensembl_ids(query)
        gene_symbols = self.extract_gene_symbols(query)
        
        # Detect intent
        intent = self.detect_intent(query)
        
        # Calculate confidence
        confidence = self.calculate_confidence(query, ensembl_ids, gene_symbols, intent)
        
        # Suggest action
        if ensembl_ids:
            suggested_action = f"run_gget_info"
        elif gene_symbols:
            suggested_action = f"search_gene_symbol"
        else:
            suggested_action = "clarify_query"
            
        return GeneQueryIntent(
            intent_type=intent,
            ensembl_ids=ensembl_ids,
            gene_symbols=gene_symbols,
            confidence=confidence,
            original_query=query,
            suggested_action=suggested_action
        )
    
    def is_gene_query(self, query: str, min_confidence: float = 0.3) -> bool:
        """Check if a query is likely about genes."""
        intent = self.process_query(query)
        return intent.confidence >= min_confidence


# Example queries for testing
EXAMPLE_QUERIES = [
    "What is ENSG00000034713?",
    "Tell me about the TP53 gene",
    "What does ENSG00000141510 do?",
    "Information about BRCA1 gene function",
    "Where is ENSG00000012048 located?",
    "What is the role of MYC gene?"
]


def test_query_processor():
    """Test the query processor with example queries."""
    processor = GeneQueryProcessor()
    
    print("🧪 Testing Gene Query Processor")
    print("=" * 50)
    
    for query in EXAMPLE_QUERIES:
        intent = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {intent.intent_type}")
        print(f"  Ensembl IDs: {intent.ensembl_ids}")
        print(f"  Gene Symbols: {intent.gene_symbols}")
        print(f"  Confidence: {intent.confidence:.2f}")
        print(f"  Action: {intent.suggested_action}")


if __name__ == "__main__":
    test_query_processor()
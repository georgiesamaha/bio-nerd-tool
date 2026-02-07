"""AI Agent integration for interpreting gget results.

This module handles communication with local AI models (Ollama)
to provide natural language explanations of gene information.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Union
import httpx
from dataclasses import dataclass


@dataclass
class AIResponse:
    """Response from AI agent."""
    
    content: str
    confidence: float
    model_used: str
    processing_time_ms: int
    success: bool
    error: Optional[str] = None


class OllamaAgent:
    """AI agent using local Ollama model for gene information interpretation."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5-coder:3b",
                 timeout: int = 60):
        """
        Initialize Ollama agent.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds (increased for local models)
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    async def _make_request(self, prompt: str) -> AIResponse:
        """Make a request to Ollama API."""
        
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_ctx": 2048  # Context window
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                processing_time = int((time.perf_counter() - start_time) * 1000)
                
                # Extract the response content
                content = result.get("response", "")
                
                if not content:
                    raise ValueError("Empty response from Ollama")
                
                return AIResponse(
                    content=content,
                    confidence=0.8,  # Default confidence for Ollama
                    model_used=self.model,
                    processing_time_ms=processing_time,
                    success=True
                )
                
        except httpx.TimeoutException as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Ollama request timed out after {self.timeout}s"
            self.logger.error(error_msg)
            
            return AIResponse(
                content="",
                confidence=0.0,
                model_used=self.model,
                processing_time_ms=processing_time,
                success=False,
                error=error_msg
            )
            
        except httpx.HTTPStatusError as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
            self.logger.error(error_msg)
            
            return AIResponse(
                content="",
                confidence=0.0,
                model_used=self.model,
                processing_time_ms=processing_time,
                success=False,
                error=error_msg
            )
            
        except Exception as e:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Ollama request failed: {str(e)}"
            self.logger.error(error_msg)
            
            return AIResponse(
                content="",
                confidence=0.0,
                model_used=self.model,
                processing_time_ms=processing_time,
                success=False,
                error=error_msg
            )
    
    async def interpret_gene_info(self, gene_id: str, gget_data: Dict[str, Any]) -> AIResponse:
        """Interpret gget gene information using AI."""
        
        # Create a structured prompt for the AI
        prompt = self._create_gene_interpretation_prompt(gene_id, gget_data)
        return await self._make_request(prompt)
    
    async def answer_gene_question(self, question: str, gene_id: str, gget_data: Dict[str, Any]) -> AIResponse:
        """Answer a specific question about a gene using AI."""
        
        prompt = self._create_question_answering_prompt(question, gene_id, gget_data)
        return await self._make_request(prompt)
    
    def _create_gene_interpretation_prompt(self, gene_id: str, gget_data: Dict[str, Any]) -> str:
        """Create a prompt for general gene information interpretation."""
        
        # Extract key information from gget data
        gene_info = self._extract_key_info(gget_data)
        
        prompt = f"""You are a bioinformatics expert helping users understand gene information.

Gene ID: {gene_id}

Raw data from bioinformatics databases:
{json.dumps(gene_info, indent=2)}

Please provide a clear, informative explanation of this gene including:

1. **Gene Overview**: What is this gene and what does it do?
2. **Key Details**: Important biological information (location, type, etc.)
3. **Function**: Known or predicted biological functions
4. **Clinical Relevance**: Any known disease associations or clinical significance (if applicable)

Keep the explanation:
- Clear and accessible to both scientists and students
- Factual and based on the provided data
- Well-structured with clear sections
- Concise but comprehensive

Do not speculate beyond what the data supports. If information is missing, mention that explicitly.

Response:"""
        
        return prompt
    
    def _create_question_answering_prompt(self, question: str, gene_id: str, gget_data: Dict[str, Any]) -> str:
        """Create a prompt for answering specific questions about a gene."""
        
        gene_info = self._extract_key_info(gget_data)
        
        prompt = f"""You are a bioinformatics expert. A user has asked a specific question about a gene.

User Question: {question}

Gene ID: {gene_id}

Available gene information:
{json.dumps(gene_info, indent=2)}

Please answer the user's question directly and accurately based on the provided data. 

Guidelines:
- Answer the specific question asked
- Use the provided gene data as your primary source
- If the data doesn't contain information to answer the question, say so clearly
- Be precise and factual
- Use clear, accessible language
- Cite specific data points when relevant

Answer:"""
        
        return prompt
    
    def _extract_key_info(self, gget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from gget response for AI processing."""
        
        if not gget_data:
            return {}
        
        # Handle pandas DataFrame-like structure
        if hasattr(gget_data, 'to_dict'):
            gget_data = gget_data.to_dict('records')
        
        if isinstance(gget_data, list) and gget_data:
            gget_data = gget_data[0]  # Take first record
        
        # Extract commonly useful fields
        key_fields = [
            'ensembl_id', 'gene_name', 'gene_symbol', 'description',
            'gene_type', 'chromosome', 'start', 'end', 'strand',
            'uniprot_id', 'canonical_transcript', 'biotype'
        ]
        
        extracted = {}
        for field in key_fields:
            if field in gget_data:
                extracted[field] = gget_data[field]
        
        # Add any other fields that might be useful
        for key, value in gget_data.items():
            if key not in extracted and value is not None:
                extracted[key] = value
        
        return extracted
    
    async def check_availability(self) -> bool:
        """Check if Ollama service is available."""
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {str(e)}")
            return False


class MockAIAgent:
    """Mock AI agent for testing when Ollama is not available."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def interpret_gene_info(self, gene_id: str, gget_data: Dict[str, Any]) -> AIResponse:
        """Mock gene information interpretation."""
        
        response = f"""**Gene Overview**
Gene ID: {gene_id}

This gene information was retrieved from public bioinformatics databases. 

**Available Data**
- Gene ID: {gene_id}
- Database records: {len(gget_data) if isinstance(gget_data, list) else 1}

**Note**: This is a simplified response. For detailed AI-powered interpretation, please ensure the Ollama service is running with the qwen2.5-coder:3b model.

To get full AI interpretation:
1. Start Ollama: `ollama serve`
2. Ensure model is available: `ollama pull qwen2.5-coder:3b`
"""
        
        return AIResponse(
            content=response,
            confidence=0.5,
            model_used="mock",
            processing_time_ms=100,
            success=True
        )
    
    async def answer_gene_question(self, question: str, gene_id: str, gget_data: Dict[str, Any]) -> AIResponse:
        """Mock question answering."""
        
        response = f"""**Question**: {question}
**Gene ID**: {gene_id}

I have retrieved the gene information from bioinformatics databases, but I need the Ollama AI service to provide a detailed interpretation.

**Raw Data Available**: Yes, gene information was successfully retrieved.

**To get AI-powered answers**:
1. Start Ollama service: `ollama serve`
2. Ensure qwen2.5-coder:3b model is available
3. The system will then provide intelligent interpretations of gene data.
"""
        
        return AIResponse(
            content=response,
            confidence=0.3,
            model_used="mock",
            processing_time_ms=50,
            success=True
        )
    
    async def check_availability(self) -> bool:
        """Mock availability check."""
        return False


async def create_ai_agent(prefer_ollama: bool = True) -> Union[OllamaAgent, MockAIAgent]:
    """Create an AI agent, preferring Ollama but falling back to mock."""
    
    if prefer_ollama:
        ollama_agent = OllamaAgent()
        if await ollama_agent.check_availability():
            return ollama_agent
    
    # Fall back to mock agent
    return MockAIAgent()


# Example usage and testing
async def test_ai_agent():
    """Test the AI agent functionality."""
    
    print("🤖 Testing AI Agent")
    print("=" * 30)
    
    agent = await create_ai_agent()
    print(f"Using agent: {type(agent).__name__}")
    
    # Mock gene data for testing
    test_gene_data = {
        'ensembl_id': 'ENSG00000034713',
        'gene_name': 'GABARAPL2',
        'description': 'GABA type A receptor associated protein like 2',
        'chromosome': '16',
        'gene_type': 'protein_coding'
    }
    
    # Test gene interpretation
    interpretation = await agent.interpret_gene_info("ENSG00000034713", test_gene_data)
    print(f"\n✅ Interpretation successful: {interpretation.success}")
    print(f"Model: {interpretation.model_used}")
    print(f"Processing time: {interpretation.processing_time_ms}ms")
    print(f"Content preview: {interpretation.content[:200]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ai_agent())
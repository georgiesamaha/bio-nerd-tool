#!/usr/bin/env python3
"""gget MCP Server - Bioinformatics query interface using the gget library."""

import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import typer
from typing_extensions import Annotated
from mcp.server.fastmcp import FastMCP
import gget
import httpx
import json
import asyncio

# Get package version
try:
    __version__ = version("gget-mcp-server")  
except PackageNotFoundError:
    __version__ = "unknown"

class TransportType(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

DEFAULT_MODEL = os.getenv("MCP_MODEL", "llama3:8b")
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3002"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

# Type hints for common return patterns
SequenceResult = Union[Dict[str, str], List[str], str]
StructureResult = Union[Dict[str, Any], str]
SearchResult = Dict[str, Any]
LocalFileResult = Dict[Literal["path", "format", "success", "error"], Any]


class GgetMCP(FastMCP):
    """Simplified gget MCP Server with essential bioinformatics tools and AI interpretation."""
    
    def __init__(
        self, 
        name: str = f"gget MCP Server v{__version__}",
        prefix: str = "gget_",
        transport_mode: str = "stdio",
        output_dir: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b",
        **kwargs
    ):
        """Initialize the gget tools with FastMCP functionality and AI integration."""
        super().__init__(name=name, **kwargs)
        
        self.prefix = prefix
        self.transport_mode = transport_mode
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "gget_output"
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Create output directory if needed
        if self.transport_mode == "stdio":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self._register_gget_tools()
    
    def _register_gget_tools(self):
        """Register gget tools with simplified interfaces."""
        
        # Natural language query tool (AI-powered)
        self.tool(name=f"{self.prefix}query")(self.natural_language_query)
        
        # Gene information and search tools  
        self.tool(name=f"{self.prefix}search")(self.search_genes)
        self.tool(name=f"{self.prefix}info")(self.get_gene_info)
        
        # Sequence tools
        if self.transport_mode == "stdio":
            self.tool(name=f"{self.prefix}seq")(self.get_sequences_local)
        else:
            self.tool(name=f"{self.prefix}seq")(self.get_sequences)
        
        # Reference genome tools
        self.tool(name=f"{self.prefix}ref")(self.get_reference)
        
        # Sequence analysis tools
        self.tool(name=f"{self.prefix}blast")(self.blast_sequence)
        self.tool(name=f"{self.prefix}blat")(self.blat_sequence)
        
        # Alignment tools
        if self.transport_mode == "stdio":
            self.tool(name=f"{self.prefix}muscle")(self.muscle_align_local)
        else:
            self.tool(name=f"{self.prefix}muscle")(self.muscle_align)
        
        # Expression analysis
        self.tool(name=f"{self.prefix}archs4")(self.archs4_expression)
        self.tool(name=f"{self.prefix}enrichr")(self.enrichr_analysis)
        
        # Protein structure
        if self.transport_mode == "stdio":
            self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure_local)
            self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict_local)
        else:
            self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure)
            self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict)
    
    # AI Agent Methods for Natural Language Processing
    
    async def _call_ai_agent(self, prompt: str, context: str = "") -> str:
        """Call the local Ollama AI agent for natural language processing."""
        try:
            full_prompt = f"""
You are a bioinformatics expert helping interpret gene and genomics data queries.

Context: {context}

User Query: {prompt}

Please provide a helpful response about the biological significance, function, or relevant information about the genes or data mentioned. Be concise but informative.
"""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response', 'No response from AI model.')
                
        except Exception as e:
            return f"AI interpretation not available: {str(e)}"
    
    async def _extract_gene_ids(self, query: str) -> List[str]:
        """Extract potential gene IDs from natural language query using AI and pattern matching."""
        
        # Use pattern matching only for high-confidence identifiers
        import re
        
        # Only trust unambiguous patterns - Ensembl IDs are never English words
        ensembl_patterns = [
            r'\b(ENSG\d{11})\b',  # Ensembl gene IDs
            r'\b(ENST\d{11})\b',  # Ensembl transcript IDs  
            r'\b(ENSP\d{11})\b',  # Ensembl protein IDs
        ]
        
        found_ids = []
        for pattern in ensembl_patterns:
            matches = re.findall(pattern, query.upper())
            found_ids.extend(matches)
        
        # For gene symbols, be more conservative - look for well-known patterns
        # but let AI validate ambiguous cases
        potential_genes = re.findall(r'\b([A-Z]{3,6}\d*[A-Z]*)\b', query.upper())
        
        pattern_ids = found_ids
        
        # If pattern matching found clear matches, return them
        if pattern_ids:
            return list(set(pattern_ids))
        
        # Otherwise, try AI extraction for more complex queries and validate potential genes
        try:
            # Combine ensembl IDs (confident) with potential gene symbols (need validation)
            candidates = list(set(found_ids + potential_genes))
            
            prompt = f"""
You are a bioinformatics expert. From this query, identify which words are actual gene identifiers vs typos or common English words.

Query: "{query}"

Potential candidates found: {candidates}

For each candidate, determine if it's:
1. An Ensembl ID (ENSG/ENST/ENSP + numbers) - always valid
2. A real gene symbol (like TP53, BRCA1, MYC) - validate these
3. A typo or common English word (like "abour", "tell", "about" "what" "is" "the" "and" "or") - exclude these

Return ONLY a JSON list of valid gene identifiers: ["ENSG00000141510", "TP53"]
If no valid genes found, return: []
"""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=10.0  # Shorter timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Try to parse JSON from the response
                response_text = result.get('response', '[]')
                try:
                    # Extract JSON from text if wrapped
                    json_match = re.search(r'\[.*?\]', response_text)
                    if json_match:
                        gene_ids = json.loads(json_match.group())
                        ai_validated = [gid for gid in gene_ids if isinstance(gid, str)]
                        return list(set(ai_validated))  # Return AI-validated results
                except:
                    pass
                    
        except Exception as e:
            # AI failed, return only the confident Ensembl IDs
            return list(set(found_ids))
        
        return list(set(found_ids))  # Fallback to Ensembl IDs only
    
    async def _get_gene_info_with_timeout(self, gene_id: str, timeout: int = 15):
        """Get gene info with timeout to handle slow Ensembl responses."""
        try:
            # Run gget.info in a thread with timeout
            def get_info():
                return gget.info(gene_id, verbose=False)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_info)
                try:
                    result = future.result(timeout=timeout)
                    return result
                except concurrent.futures.TimeoutError:
                    return {"error": f"Timeout: Ensembl server too slow (>{timeout}s)"}
        except Exception as e:
            return {"error": f"Failed to get gene info: {str(e)}"}
    
    # Tool Implementations
    
    async def natural_language_query(
        self, 
        query: str,
        include_ai_interpretation: bool = True
    ) -> Dict[str, Any]:
        """Process a natural language query about genes or genomics data.
        
        This tool automatically detects gene identifiers in natural language
        and retrieves relevant information, with optional AI-powered interpretation.
        """
        results = {
            "original_query": query,
            "gene_ids_found": [],
            "gene_information": {},
            "ensembl_ids_resolved": {},
            "ai_interpretation": None,
            "suggestions": []
        }
        
        # Extract potential gene IDs
        gene_ids = await self._extract_gene_ids(query)
        results["gene_ids_found"] = gene_ids
        
        # Get information for found gene IDs
        gene_info_results = {}
        ensembl_resolved = {}
        
        for gene_id in gene_ids[:3]:  # Limit to 3 genes for faster processing
            try:
                # Check if it's already an Ensembl ID
                if gene_id.startswith('ENS'):
                    # Direct Ensembl ID with timeout
                    info = await self._get_gene_info_with_timeout(gene_id)
                    if isinstance(info, dict) and 'error' in info:
                        gene_info_results[gene_id] = info
                    elif info is not None and not (isinstance(info, list) and len(info) == 0):
                        gene_info_results[gene_id] = info
                        ensembl_resolved[gene_id] = gene_id
                    else:
                        gene_info_results[gene_id] = {"error": "No information found for Ensembl ID"}
                else:
                    # Gene symbol - need to search for Ensembl ID first
                    try:
                        # Limit search results to improve performance (genes usually in top 20)
                        search_results = gget.search(gene_id, species="homo_sapiens", limit=20)
                        if search_results is not None and len(search_results) > 0:
                            # Look for exact gene symbol match first
                            ensembl_id = None
                            first_result = None
                            
                            if hasattr(search_results, 'iloc'):
                                # Try to find exact match by gene symbol
                                for idx in range(len(search_results)):
                                    row = search_results.iloc[idx]
                                    gene_name = row.get('gene_name', '').upper()
                                    gene_symbol = row.get('gene_symbol', '').upper() 
                                    
                                    # Check for exact match (case insensitive)
                                    if (gene_name == gene_id.upper() or 
                                        gene_symbol == gene_id.upper() or
                                        gene_id.upper() in [gene_name, gene_symbol]):
                                        ensembl_id = row.get('ensembl_id')
                                        first_result = row
                                        break
                                
                                # If no exact match, try to find best match before falling back
                                if ensembl_id is None:
                                    # Look for partial matches or containing patterns
                                    for idx in range(len(search_results)):
                                        row = search_results.iloc[idx]
                                        gene_name = row.get('gene_name', '').upper()
                                        
                                        # Check if gene_name contains our search term exactly
                                        if gene_name == gene_id.upper():
                                            ensembl_id = row.get('ensembl_id')
                                            first_result = row
                                            break
                                    
                                    # Final fallback to first result only if still no match
                                    if ensembl_id is None:
                                        first_result = search_results.iloc[0]
                                        ensembl_id = first_result.get('ensembl_id')
                                
                                if ensembl_id:
                                    ensembl_resolved[gene_id] = ensembl_id
                                    # Get detailed info using Ensembl ID
                                    info = gget.info(ensembl_id, verbose=False)
                                    if info is not None and not (isinstance(info, list) and len(info) == 0):
                                        gene_info_results[gene_id] = {
                                            "gene_symbol": gene_id,
                                            "ensembl_id": ensembl_id,
                                            "search_result": first_result.to_dict(),
                                            "detailed_info": info
                                        }
                                    else:
                                        gene_info_results[gene_id] = {
                                            "gene_symbol": gene_id,
                                            "ensembl_id": ensembl_id,
                                            "search_result": first_result.to_dict(),
                                            "error": "Found Ensembl ID but no detailed info available"
                                        }
                                else:
                                    gene_info_results[gene_id] = {
                                        "error": "Search result found but no Ensembl ID field",
                                        "search_result": first_result.to_dict() if first_result is not None else {}
                                    }
                            else:
                                gene_info_results[gene_id] = {"error": "Search results in unexpected format"}
                        else:
                            gene_info_results[gene_id] = {"error": "No search results found"}
                    except Exception as search_error:
                        gene_info_results[gene_id] = {"error": f"Search failed: {str(search_error)}"}
                        
            except Exception as e:
                gene_info_results[gene_id] = {"error": str(e)}
        
        results["gene_information"] = gene_info_results
        results["ensembl_ids_resolved"] = ensembl_resolved
        
        # Add AI interpretation if requested and we have data
        if include_ai_interpretation and (gene_ids or gene_info_results):
            context_info = ""
            if gene_info_results:
                context_info = f"Gene data retrieved: {json.dumps(gene_info_results, default=str)[:1000]}..."
            
            ai_response = await self._call_ai_agent(query, context_info)
            results["ai_interpretation"] = ai_response
        
        # Add suggestions if no gene IDs found
        if not gene_ids:
            results["suggestions"] = [
                "Try using specific gene symbols (e.g., 'TP53', 'BRCA1')",
                "Include Ensembl IDs (e.g., 'ENSG00000141510')",
                "Use the 'search' tool to find genes by keywords"
            ]
        
        return results
    
    # Simplified tool implementations
    
    async def search_genes(
        self, 
        search_terms: Union[str, List[str]], 
        species: str = "homo_sapiens",
        limit: Optional[int] = None
    ) -> SearchResult:
        """Search for genes using gene symbols, names, or synonyms.
        
        Use this tool FIRST when you have gene names/symbols and need Ensembl IDs.
        
        Args:
            search_terms: Gene symbols or names (e.g., 'TP53' or ['TP53', 'BRCA1'])
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus')
            limit: Maximum results (default: 15 for lists, 10 for single terms)
        
        Returns:
            SearchResult: DataFrame with Ensembl IDs and descriptions
        """
        import gget
        
        # Set reasonable limits
        if limit is None:
            if isinstance(search_terms, list):
                limit = min(15, len(search_terms) * 3)
            else:
                limit = 10
        
        result = gget.search(
            searchwords=search_terms,
            species=species,
            limit=limit
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_gene_info(
        self, 
        ensembl_ids: Union[str, List[str]],
        ncbi: bool = True,
        uniprot: bool = True
    ) -> Dict[str, Any]:
        """Get detailed gene information using Ensembl IDs.
        
        Args:
            ensembl_ids: Ensembl gene IDs (e.g., 'ENSG00000141510')
            ncbi: Include NCBI data
            uniprot: Include UniProt data
        
        Returns:
            Dict with gene metadata from multiple databases
        """
        import gget
        
        result = gget.info(
            ens_ids=ensembl_ids,
            ncbi=ncbi,
            uniprot=uniprot,
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_sequences(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False
    ) -> SequenceResult:
        """Fetch gene/protein sequences.
        
        Args:
            ensembl_ids: Ensembl gene IDs
            translate: If True, returns protein sequences; if False, DNA sequences
        
        Returns:
            List of sequences in FASTA format
        """
        import gget
        
        result = gget.seq(
            ens_ids=ensembl_ids,
            translate=translate,
            verbose=False
        )
        return result

    async def get_reference(
        self, 
        species: str = "homo_sapiens",
        which: str = "all"
    ) -> Dict[str, Any]:
        """Get reference genome URLs from Ensembl.
        
        Args:
            species: Species (e.g., "homo_sapiens", "mus_musculus")
            which: Data type ('gtf', 'dna', 'cdna', 'pep', or 'all')
        
        Returns:
            Dictionary with download URLs and metadata
        """
        import gget
        
        result = gget.ref(
            species=species,
            which=which,
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blast_sequence(
        self, 
        sequence: str,
        program: str = "default",
        database: str = "default",
        limit: int = 10
    ) -> Dict[str, Any]:
        """BLAST sequence against public databases.
        
        Args:
            sequence: Nucleotide or amino acid sequence
            program: BLAST program (auto-detected by default)
            database: Database (auto-detected by default)
            limit: Maximum number of results
        
        Returns:
            BLAST results with alignments and scores
        """
        import gget
        
        result = gget.blast(
            sequence=sequence,
            program=program,
            database=database,
            limit=limit,
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blat_sequence(
        self, 
        sequence: str,
        assembly: str = "human"
    ) -> Dict[str, Any]:
        """BLAT sequence against genome assembly.
        
        Args:
            sequence: Nucleotide or amino acid sequence
            assembly: Genome assembly ('human', 'mouse', etc.)
        
        Returns:
            BLAT results with genomic coordinates
        """
        import gget
        
        result = gget.blat(
            sequence=sequence,
            assembly=assembly,
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def muscle_align(
        self, 
        sequences: Union[List[str], str]
    ) -> Optional[str]:
        """Align multiple sequences using MUSCLE.
        
        Args:
            sequences: List of sequences or path to FASTA file
        
        Returns:
            Aligned sequences in FASTA format
        """
        import gget
        
        result = gget.muscle(
            fasta=sequences,
            verbose=False
        )
        return result

    async def archs4_expression(
        self, 
        gene: str,
        which: str = "correlation",
        species: str = "human"
    ) -> Dict[str, Any]:
        """Find gene correlations or tissue expression using ARCHS4.
        
        Args:
            gene: Gene symbol (e.g., 'TP53')
            which: 'correlation' or 'tissue'
            species: 'human' or 'mouse'
        
        Returns:
            Correlation data or tissue expression atlas
        """
        import gget
        
        result = gget.archs4(
            gene=gene,
            which=which,
            species=species,
            gene_count=20,  # Limit for manageable results
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def enrichr_analysis(
        self, 
        genes: List[str],
        database: str = "KEGG_2021_Human",
        species: str = "human"
    ) -> Dict[str, Any]:
        """Perform pathway enrichment analysis.
        
        Args:
            genes: List of gene symbols
            database: Enrichr database name  
            species: Target species
        
        Returns:
            Enrichment results with pathways and statistics
        """
        import gget
        
        result = gget.enrichr(
            genes=genes,
            database=database,
            species=species,
            verbose=False
        )
        return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_pdb_structure(
        self, 
        pdb_id: str,
        resource: str = "pdb"
    ) -> StructureResult:
        """Get protein structure from PDB.
        
        Args:
            pdb_id: PDB ID (e.g., '7S7U')
            resource: 'pdb' for structure, 'entry' for metadata
        
        Returns:
            PDB structure or metadata
        """
        import gget
        
        result = gget.pdb(
            pdb_id=pdb_id,
            resource=resource,
            verbose=False
        )
        return result

    async def alphafold_predict(
        self, 
        sequence: Union[str, List[str]]
    ) -> StructureResult:
        """Predict protein structure using AlphaFold.
        
        Args:
            sequence: Amino acid sequence(s)
        
        Returns:
            Structure prediction files and confidence scores
        """
        import gget
        
        result = gget.alphafold(
            sequence=sequence,
            verbose=False
        )
        return result

    # Local file handling methods for stdio mode
    
    def _save_to_local_file(
        self, 
        data: Any, 
        format_type: str, 
        output_path: Optional[str] = None,
        default_prefix: str = "gget_output"
    ) -> LocalFileResult:
        """Save data to local file."""
        from uuid import uuid4
        import json
        
        # Map format types to extensions
        extensions = {
            'fasta': '.fasta',
            'pdb': '.pdb', 
            'json': '.json',
            'txt': '.txt'
        }
        
        extension = extensions.get(format_type, '.txt')
        
        if output_path is None:
            base_name = f"{default_prefix}_{str(uuid4())[:8]}"
            file_path = self.output_dir / f"{base_name}{extension}"
        else:
            file_path = Path(output_path)
            if not file_path.suffix:
                file_path = file_path.with_suffix(extension)
        
        try:
            with open(file_path, 'w') as f:
                if format_type == 'fasta':
                    self._write_fasta_file(data, f)
                elif format_type == 'json':
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))
            
            return {
                "path": str(file_path.absolute()),
                "format": format_type,
                "success": True
            }
        except Exception as e:
            return {
                "path": None,
                "format": format_type,
                "success": False,
                "error": str(e)
            }
    
    def _write_fasta_file(self, data: Any, file_handle):
        """Write sequence data in FASTA format."""
        if isinstance(data, dict):
            for seq_id, sequence in data.items():
                file_handle.write(f">{seq_id}\n")
                # Write with line breaks every 80 chars
                for i in range(0, len(sequence), 80):
                    file_handle.write(f"{sequence[i:i+80]}\n")
        elif isinstance(data, list):
            for i in range(0, len(data), 2):
                if i + 1 < len(data):
                    header = data[i] if data[i].startswith('>') else f">{data[i]}"
                    sequence = data[i + 1]
                    file_handle.write(f"{header}\n")
                    for j in range(0, len(sequence), 80):
                        file_handle.write(f"{sequence[j:j+80]}\n")
        else:
            file_handle.write(str(data))

    # Local file versions of tools
    
    async def get_sequences_local(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False,
        output_path: Optional[str] = None
    ) -> LocalFileResult:
        """Get sequences and save to local file."""
        
        # Get sequences using standard method
        result = await self.get_sequences(ensembl_ids, translate)
        
        # Save to file
        ensembl_list = ensembl_ids if isinstance(ensembl_ids, list) else [ensembl_ids]
        prefix = f"sequences_{'_'.join(ensembl_list[:3])}{'_protein' if translate else '_dna'}"
        
        return self._save_to_local_file(result, 'fasta', output_path, prefix)

    async def muscle_align_local(
        self, 
        sequences: Union[List[str], str],
        output_path: Optional[str] = None
    ) -> LocalFileResult:
        """Align sequences and save to local file."""
        from uuid import uuid4
        
        # Generate output path if not provided
        if output_path is None:
            base_name = f"muscle_alignment_{str(uuid4())[:8]}"
            file_path = self.output_dir / f"{base_name}.fasta"
        else:
            file_path = Path(output_path)
        
        try:
            import gget
            # Use gget.muscle with direct file output
            result = gget.muscle(
                fasta=sequences,
                out=str(file_path),
                verbose=False
            )
            
            return {
                "path": str(file_path.absolute()),
                "format": "fasta",
                "success": True
            }
        except Exception as e:
            return {
                "path": None,
                "format": "fasta", 
                "success": False,
                "error": str(e)
            }

    async def get_pdb_structure_local(
        self, 
        pdb_id: str,
        resource: str = "pdb",
        output_path: Optional[str] = None
    ) -> LocalFileResult:
        """Get PDB structure and save to local file."""
        
        # Get structure using standard method
        result = await self.get_pdb_structure(pdb_id, resource)
        
        # Save to file
        prefix = f"structure_{pdb_id}_{resource}"
        return self._save_to_local_file(result, 'pdb', output_path, prefix)

    async def alphafold_predict_local(
        self, 
        sequence: Union[str, List[str]],
        output_path: Optional[str] = None
    ) -> LocalFileResult:
        """Predict structure and save to local file."""
        
        # Get prediction using standard method  
        result = await self.alphafold_predict(sequence)
        
        # Save to file
        from uuid import uuid4
        prefix = f"alphafold_prediction_{str(uuid4())[:8]}"
        return self._save_to_local_file(result, 'pdb', output_path, prefix)


def create_app(
    transport_mode: str = "stdio", 
    output_dir: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    model_name: str = "llama3:8b"
):
    """Create and configure the gget MCP application."""
    return GgetMCP(
        transport_mode=transport_mode, 
        output_dir=output_dir,
        ollama_url=ollama_url,
        model_name=model_name
    )


# CLI application setup
cli_app = typer.Typer(help="gget MCP Server CLI")

@cli_app.command()
def server(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    transport: Annotated[str, typer.Option(help="Transport type: stdio, streamable-http, or sse")] = DEFAULT_TRANSPORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files (stdio mode)")] = None,
    ollama_url: Annotated[str, typer.Option(help="Ollama server URL for AI integration")] = "http://localhost:11434",
    model_name: Annotated[str, typer.Option(help="Ollama model name for AI integration")] = "llama3:8b"
):
    """Run the gget MCP server."""
    
    # Validate transport
    if transport not in ["stdio", "streamable-http", "sse"]:
        typer.echo(f"Invalid transport: {transport}. Must be one of: stdio, streamable-http, sse")
        raise typer.Exit(1)
    
    app = create_app(
        transport_mode=transport, 
        output_dir=output_dir,
        ollama_url=ollama_url,
        model_name=model_name
    )
    
    # Run with appropriate transport
    if transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport=transport, host=host, port=port)

@cli_app.command(name="stdio")
def stdio_mode(
    ollama_url: Annotated[str, typer.Option(help="Ollama server URL for AI integration")] = "http://localhost:11434",
    model_name: Annotated[str, typer.Option(help="Ollama model name for AI integration")] = "llama3:8b"
):
    """Run the gget MCP server in stdio mode."""
    app = create_app(transport_mode="stdio", ollama_url=ollama_url, model_name=model_name)
    app.run(transport="stdio")

@cli_app.command(name="http")
def http_mode(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files")] = None,
    ollama_url: Annotated[str, typer.Option(help="Ollama server URL for AI integration")] = "http://localhost:11434",
    model_name: Annotated[str, typer.Option(help="Ollama model name for AI integration")] = "llama3:8b"
):
    """Run the gget MCP server in HTTP mode."""
    app = create_app(
        transport_mode="streamable-http", 
        output_dir=output_dir,
        ollama_url=ollama_url,
        model_name=model_name
    )
    app.run(transport="streamable-http", host=host, port=port)

@cli_app.command(name="sse")
def sse_mode(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files")] = None,
    ollama_url: Annotated[str, typer.Option(help="Ollama server URL for AI integration")] = "http://localhost:11434",
    model_name: Annotated[str, typer.Option(help="Ollama model name for AI integration")] = "llama3:8b"
):
    """Run the gget MCP server in Server-Sent Events mode."""
    app = create_app(
        transport_mode="sse", 
        output_dir=output_dir,
        ollama_url=ollama_url,
        model_name=model_name
    )
    app.run(transport="sse", host=host, port=port)

@cli_app.command(name="query")
def query_mode(
    query: Annotated[Optional[str], typer.Argument(help="Your bioinformatics query (optional - if not provided, starts interactive mode)")] = None,
    ollama_url: Annotated[str, typer.Option(help="Ollama server URL for AI integration")] = "http://localhost:11434",
    model_name: Annotated[str, typer.Option(help="Ollama model name for AI integration")] = "llama3:8b",
    include_ai: Annotated[bool, typer.Option("--ai/--no-ai", help="Include AI interpretation")] = True
):
    """Ask a direct bioinformatics question and get results."""
    
    # If no query provided, start interactive mode
    if query is None:
        typer.echo("🧬 bio-nerd: What can I help you with?")
        try:
            query = input()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n👋 Goodbye!")
            return
        if not query.strip():
            typer.echo("👋 Goodbye!")
            return
    
    async def run_query():
        app = create_app(transport_mode="stdio", ollama_url=ollama_url, model_name=model_name)
        
        typer.echo(f"\n🔬 Processing: {query}")
        typer.echo("-" * 50)
        
        try:
            result = await app.natural_language_query(query, include_ai_interpretation=include_ai)
            
            # Display gene IDs found
            gene_ids = result.get('gene_ids_found', [])
            if gene_ids:
                typer.echo(f"🎯 Gene IDs detected: {', '.join(gene_ids)}")
            else:
                typer.echo("🤔 No gene identifiers detected in query")
            
            # Display resolved Ensembl mappings
            ensembl_resolved = result.get('ensembl_ids_resolved', {})
            if ensembl_resolved:
                typer.echo("🔗 Ensembl ID mappings:")
                for symbol, ensembl_id in ensembl_resolved.items():
                    typer.echo(f"   {symbol} → {ensembl_id}")
            
            # Display gene information
            gene_info = result.get('gene_information', {})
            if gene_info:
                typer.echo("\n📊 Gene Information:")
                for gene_id, info in gene_info.items():
                    typer.echo(f"\n• {gene_id}:")
                    if isinstance(info, dict):
                        if isinstance(info, dict):
                            if 'error' in info:
                                typer.echo(f"   ❌ {info['error']}")
                            elif 'detailed_info' in info:
                                detailed = info['detailed_info']
                                if hasattr(detailed, 'iloc') and len(detailed) > 0:
                                    # DataFrame result from gget.info
                                    first_row = detailed.iloc[0]
                                    gene_name = first_row.get('primary_gene_name', 'Unknown')
                                    description = first_row.get('ensembl_description', first_row.get('protein_names', 'Unknown'))
                                    typer.echo(f"   ✅ {gene_name}")
                                    typer.echo(f"   📝 {description}")
                                elif isinstance(detailed, dict):
                                    gene_name = detailed.get('primary_gene_name', detailed.get('gene_name', 'Unknown'))
                                    description = detailed.get('ensembl_description', detailed.get('protein_names', 'Unknown'))
                                    typer.echo(f"   ✅ {gene_name}")
                                    typer.echo(f"   📝 {description}")
                                else:
                                    typer.echo(f"   ✅ Data retrieved (type: {type(detailed).__name__})")
                            elif 'ensembl_id' in info:
                                typer.echo(f"   🔗 Resolved to Ensembl ID: {info['ensembl_id']}")
                            else:
                                typer.echo(f"   📋 {str(info)[:100]}...")
                        else:
                            # Direct result from gget.info
                            if hasattr(info, 'iloc') and len(info) > 0:
                                first_row = info.iloc[0]
                                gene_name = first_row.get('primary_gene_name', 'Unknown')
                                description = first_row.get('ensembl_description', first_row.get('protein_names', 'Unknown'))
                                typer.echo(f"   ✅ {gene_name}")
                                typer.echo(f"   📝 {description}")
                            else:
                                typer.echo(f"   📋 {str(info)[:100]}...")
            
            # Display AI interpretation
            ai_response = result.get('ai_interpretation')
            if ai_response and include_ai:
                typer.echo("\n🧠 AI Interpretation:")
                typer.echo(ai_response)
            
            # Display suggestions if no results
            suggestions = result.get('suggestions', [])
            if suggestions:
                typer.echo("\n💡 Suggestions:")
                for suggestion in suggestions:
                    typer.echo(f"   • {suggestion}")
                    
        except Exception as e:
            typer.echo(f"💥 Error: {e}")
    
    asyncio.run(run_query())

def main():
    """Entry point for the CLI application."""
    cli_app()

if __name__ == "__main__":
    main()
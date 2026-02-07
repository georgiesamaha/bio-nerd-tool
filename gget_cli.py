#!/usr/bin/env python3
"""
Command-line interface for gget-MCP.

Simple CLI tool to query gene information with safety controls.
Usage: python3 gget_cli.py <gene_id>
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gget_mcp.tools.gget_info import GgetInfoTool
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you've installed dependencies: pip install -e .")
    sys.exit(1)


async def query_gene(gene_id: str, confidence_level: str = "standard"):
    """Query gene information using gget-MCP."""
    
    # Quick network connectivity check
    print("🔍 Checking network connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen('https://httpbin.org/ip', timeout=3)
        print("✅ Network connectivity OK")
    except Exception as e:
        print(f"❌ Network connectivity failed: {e}")
        print("🌐 This explains why gget queries fail")
        print(f"💡 Use offline mode instead: python3 offline_gget.py {gene_id}")
        return False
    
    tool = GgetInfoTool()
    
    print(f"🧬 Querying gene: {gene_id}")
    print("=" * 50)
    
    try:
        result = await tool.execute({
            "gene_id": gene_id,
            "confidence_level": confidence_level
        })
        
        if result.get("success"):
            data = result.get("data", {})
            metadata = result.get("epistemic_metadata", {})
            provenance = result.get("provenance_metadata", {})
            
            # Display gene information
            print(f"✅ Gene Name: {data.get('gene_name', 'Unknown')}")
            print(f"📝 Description: {data.get('description', 'N/A')}")
            print(f"🧬 Biotype: {data.get('biotype', 'N/A')}")
            print(f"📍 Location: Chr {data.get('chromosome', 'N/A')}")
            
            if data.get('start') and data.get('end'):
                print(f"🔗 Position: {data.get('start')}-{data.get('end')}")
            
            print(f"📊 Strand: {data.get('strand', 'N/A')}")
            
            # Display confidence information
            print("\\n" + "=" * 50)
            print("🎯 CONFIDENCE ASSESSMENT")
            print("=" * 50)
            confidence = metadata.get('confidence_level', 'unknown')
            score = metadata.get('confidence_score', 0)
            print(f"📈 Confidence Level: {confidence.upper()}")
            print(f"📊 Confidence Score: {score:.2f}")
            
            uncertainty = metadata.get('uncertainty_sources', [])
            if uncertainty:
                print(f"⚠️  Uncertainty Sources: {', '.join(uncertainty)}")
            
            # Display source information  
            print("\\n" + "=" * 50)
            print("📚 SOURCE ATTRIBUTION")
            print("=" * 50)
            primary_source = provenance.get('primary_source', 'Unknown')
            print(f"🏛️  Primary Source: {primary_source}")
            
            if provenance.get('data_version'):
                print(f"📅 Data Version: {provenance.get('data_version')}")
            
            query_time = provenance.get('query_timestamp', 'Unknown')
            print(f"⏰ Query Time: {query_time}")
            
            return True
            
        else:
            # Handle errors
            error_info = result.get("failure_mode", {})
            refusal = result.get("refusal_template", {})
            error_context = result.get("error_context", {})
            
            print(f"❌ Query Failed")
            
            # Show user-friendly message
            user_message = error_info.get('user_message') or refusal.get('message', 'Unknown error')
            print(f"💬 Reason: {user_message}")
            
            # Show technical details if available
            if error_context.get('stack_trace'):
                print(f"\\n🔧 Technical Details:")
                print(error_context['stack_trace'][:500] + "..." if len(error_context['stack_trace']) > 500 else error_context['stack_trace'])
            
            suggestions = error_info.get('recovery_suggestions', []) or refusal.get('alternative_suggestions', [])
            if suggestions:
                print("\\n💡 Suggestions:")
                for suggestion in suggestions:
                    print(f"   • {suggestion}")
            
            return False
            
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"🔧 Error Type: {type(e).__name__}")
        
        # Diagnostic information
        error_str = str(e).lower()
        if "timeout" in error_str or "connection" in error_str:
            print("🌐 NETWORK ISSUE: Cannot connect to gene databases")
            print("   → Your workspace may not have internet access")
            print("   → Try: python3 offline_gget.py TP53 (works without internet)")
        elif "ssl" in error_str or "certificate" in error_str:
            print("🔒 SSL ISSUE: Certificate validation failed")
            print("   → SSL/TLS configuration problem")
        elif "dns" in error_str or "resolve" in error_str:
            print("🔍 DNS ISSUE: Cannot resolve gene database hostnames") 
        elif "permission" in error_str or "denied" in error_str:
            print("🚫 PERMISSION ISSUE: Network access blocked")
        else:
            print("❓ UNKNOWN ISSUE:")
            import traceback
            traceback.print_exc()
            
        print(f"\n💡 Workaround: python3 offline_gget.py {gene_id}")
        return False


def print_usage():
    """Print usage instructions."""
    
    print("🧬 gget-MCP Command Line Tool")
    print("=" * 40)
    print("\\nUsage:")
    print("  python3 gget_cli.py <gene_id> [confidence_level]")
    print("\\nExamples:")
    print("  python3 gget_cli.py TP53")
    print("  python3 gget_cli.py ENSG00000157764")
    print("  python3 gget_cli.py BRCA1 high")
    print("\\nConfidence Levels:")
    print("  • low      - Accept lower quality data")  
    print("  • standard - Default quality threshold")
    print("  • high     - Only high-confidence data")
    print("\\nSupported Gene IDs:")
    print("  • Gene symbols: TP53, BRCA1, EGFR")
    print("  • Ensembl IDs: ENSG00000141510") 
    print("  • NCBI IDs: Various formats")


async def main():
    """Main CLI entry point."""
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    gene_id = sys.argv[1]
    confidence_level = sys.argv[2] if len(sys.argv) > 2 else "standard"
    
    # Validate confidence level
    valid_levels = ["low", "standard", "high"]
    if confidence_level not in valid_levels:
        print(f"❌ Invalid confidence level: {confidence_level}")
        print(f"Valid options: {', '.join(valid_levels)}")
        sys.exit(1)
    
    success = await query_gene(gene_id, confidence_level)
    
    if success:
        print("\\n✅ Query completed successfully")
        sys.exit(0)
    else:
        print("\\n❌ Query failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n\\n👋 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
gget-MCP Command Line Interface

A beautiful CLI for running the gget-MCP server with local AI capabilities.
"""

import asyncio
import sys
import time
from typing import Optional
import subprocess
import requests


def print_banner():
    """Print the gget-MCP banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                 gget-MCP                                     ║
║                     Bioinformatics Query Server                             ║
║                   🧬 Powered by Local AI (qwen2.5-coder:3b)                ║  
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_system_status() -> dict:
    """Check system requirements and status."""
    
    print("🔍 Checking System Status...")
    status = {}
    
    # Check Python imports
    try:
        import gget
        status['gget'] = {'available': True, 'version': getattr(gget, '__version__', 'unknown')}
        print("  ✅ gget library: Available")
    except ImportError:
        status['gget'] = {'available': False, 'version': None}
        print("  ❌ gget library: NOT AVAILABLE")
    
    try:
        from mcp.server.fastmcp import FastMCP
        status['mcp'] = {'available': True}
        print("  ✅ MCP library: Available")
    except ImportError:
        status['mcp'] = {'available': False}
        print("  ❌ MCP library: NOT AVAILABLE")
    
    # Check Ollama service
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            qwen_available = any('qwen2.5-coder:3b' in model['name'] for model in models)
            status['ollama'] = {
                'available': True, 
                'qwen_model': qwen_available,
                'models': [model['name'] for model in models]
            }
            print(f"  ✅ Ollama service: Running")
            if qwen_available:
                print(f"  ✅ qwen2.5-coder:3b: Available")
            else:
                print(f"  ⚠️  qwen2.5-coder:3b: Not found")
        else:
            status['ollama'] = {'available': False}
            print("  ❌ Ollama service: Not responding")
    except Exception:
        status['ollama'] = {'available': False}
        print("  ❌ Ollama service: Not running")
    
    # Check gget-MCP server imports
    try:
        from gget_mcp.server import create_mcp_server
        status['gget_mcp_server'] = {'available': True}
        print("  ✅ gget-MCP server: Ready")
    except ImportError as e:
        status['gget_mcp_server'] = {'available': False, 'error': str(e)}
        print(f"  ❌ gget-MCP server: Import failed")
    
    return status


def print_status_summary(status: dict):
    """Print a summary of system status."""
    
    print("\n" + "="*80)
    print("📋 SYSTEM STATUS SUMMARY")
    print("="*80)
    
    all_good = True
    
    # Core requirements
    if not status.get('gget', {}).get('available'):
        print("❌ Missing: gget library (run: pip install gget)")
        all_good = False
    
    if not status.get('mcp', {}).get('available'):
        print("❌ Missing: MCP library (run: pip install mcp)")
        all_good = False
        
    if not status.get('gget_mcp_server', {}).get('available'):
        print("❌ Missing: gget-MCP server (run: pip install -e .)")
        all_good = False
    
    # AI capabilities
    ollama_status = status.get('ollama', {})
    if not ollama_status.get('available'):
        print("⚠️  Ollama not running (AI features disabled)")
        print("   To enable AI: ollama serve")
    elif not ollama_status.get('qwen_model'):
        print("⚠️  qwen2.5-coder:3b model missing")
        print("   To install: ollama pull qwen2.5-coder:3b")
    else:
        print("🤖 AI capabilities: Fully operational")
    
    if all_good and ollama_status.get('available'):
        print("🎉 All systems ready! You can start the MCP server.")
    elif all_good:
        print("✅ Basic functionality ready (AI features require Ollama)")
    else:
        print("⚠️  Some components need installation before starting")
        
    print("="*80)


def show_capabilities(status: dict):
    """Show what the system can do."""
    
    print("\n🚀 CAPABILITIES")
    print("-" * 50)
    
    # Basic capabilities
    if status.get('gget', {}).get('available'):
        print("📊 Gene Information Queries:")
        print("   • Retrieve gene data from Ensembl, NCBI, UniProt")
        print("   • Command: gget_info(gene_id='ENSG00000034713')")
    
    # AI capabilities  
    ollama_status = status.get('ollama', {})
    if ollama_status.get('available') and ollama_status.get('qwen_model'):
        print("\n🤖 AI-Powered Natural Language Queries:")
        print("   • Ask: 'What is ENSG00000034713?'")
        print("   • Ask: 'Tell me about the TP53 gene'")
        print("   • Ask: 'What does BRCA1 do?'")
        print("   • Command: ask_about_gene(query='your question')")
    elif ollama_status.get('available'):
        print("\n⚠️  AI Features (needs qwen2.5-coder:3b model):")
        print("   • Natural language gene queries")
        print("   • Run: ollama pull qwen2.5-coder:3b")
    else:
        print("\n💤 AI Features (needs Ollama service):")
        print("   • Natural language gene queries") 
        print("   • Run: ollama serve")
    
    print("\n🔒 Safety Features:")
    print("   • Read-only access to public databases")
    print("   • Input validation and sanitization")
    print("   • Complete audit trail and provenance tracking")
    print("   • No external API dependencies (when using local AI)")


def show_usage_examples():
    """Show usage examples."""
    
    print("\n💡 USAGE EXAMPLES")
    print("-" * 50)
    print("After starting the MCP server, you can use these tools:")
    print()
    
    print("1️⃣  Gene Information (Direct):")
    print("   gget_info(gene_id='ENSG00000034713')")
    print()
    
    print("2️⃣  Natural Language Queries:")
    print("   ask_about_gene(query='What is ENSG00000034713?')")
    print("   ask_about_gene(query='Tell me about TP53')")
    print("   ask_about_gene(query='What does BRCA1 do?')")
    print()
    
    print("3️⃣  System Information:")
    print("   system_capabilities()  # Show available tools")
    print("   system_status()        # Check system health")


def start_server():
    """Start the gget-MCP server."""
    
    print("\n🚀 Starting gget-MCP Server...")
    print("-" * 50)
    print("Server Mode: stdio (MCP protocol)")
    print("AI Model: Local qwen2.5-coder:3b (if available)")
    print("Status: Ready for client connections")
    print("-" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        from gget_mcp.server import main_sync
        main_sync()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        sys.exit(1)


def show_menu(status: dict) -> str:
    """Show interactive menu and get user choice."""
    
    print("\n📋 MENU OPTIONS")
    print("-" * 30)
    print("1. Start MCP Server")
    print("2. Check System Status")  
    print("3. Show Capabilities")
    print("4. Usage Examples")
    print("5. Setup Help")
    print("q. Quit")
    print()
    
    choice = input("Choose an option [1-5, q]: ").strip().lower()
    return choice


def show_setup_help():
    """Show setup and troubleshooting help."""
    
    print("\n🛠️  SETUP HELP")
    print("="*60)
    
    print("\n📦 Installation:")
    print("   pip install gget mcp")
    print("   pip install -e .  # Install gget-MCP")
    
    print("\n🤖 Ollama Setup (for AI features):")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ollama serve &")
    print("   ollama pull qwen2.5-coder:3b")
    
    print("\n🔧 Quick Start:")
    print("   1. Run this CLI: python3 gget_mcp_cli.py")
    print("   2. Check status (option 2)")
    print("   3. Start server (option 1)")
    
    print("\n📱 MCP Client Setup:")
    print("   • Claude Desktop: Add server config to settings")
    print("   • Other MCP clients: Use stdio transport")
    
    print("\n🐛 Troubleshooting:")
    print("   • Module not found: pip install -e .")
    print("   • Ollama not working: Check 'ollama serve' is running")
    print("   • Network errors: Use offline mode")


def main():
    """Main CLI entry point."""
    
    print_banner()
    
    # Initial system check
    status = check_system_status()
    print_status_summary(status)
    
    # Interactive menu
    while True:
        choice = show_menu(status)
        
        if choice == '1':
            # Check if we can start server
            if not status.get('gget_mcp_server', {}).get('available'):
                print("\n❌ Cannot start server - missing dependencies")
                print("Run 'pip install -e .' first")
                continue
            start_server()
            break
            
        elif choice == '2':
            status = check_system_status()
            print_status_summary(status)
            
        elif choice == '3':
            show_capabilities(status)
            
        elif choice == '4':
            show_usage_examples()
            
        elif choice == '5':
            show_setup_help()
            
        elif choice == 'q':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
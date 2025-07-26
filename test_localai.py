#!/usr/bin/env python3
"""
Test script to verify LocalAI integration
"""

import os
import sys
sys.path.append('/home/ubuntu/job_application_agent_free/modified')

from src.utils.localai_client import get_localai_client, create_localai_llm
from src.utils.llm_provider import get_llm_model

def test_localai_client():
    """Test LocalAI client functionality"""
    print("Testing LocalAI client...")
    
    # Test client creation
    client = get_localai_client()
    if client:
        print("✅ LocalAI client created successfully")
        
        # Test availability
        if client.is_available():
            print("✅ LocalAI server is available")
            
            # Test model listing
            models = client.list_models()
            print(f"📋 Available models: {models}")
            
        else:
            print("❌ LocalAI server is not available")
    else:
        print("❌ Failed to create LocalAI client")

def test_llm_provider():
    """Test LLM provider with LocalAI"""
    print("\nTesting LLM provider...")
    
    # Set environment for LocalAI
    os.environ['USE_LOCALAI'] = 'true'
    os.environ['LOCALAI_URL'] = 'http://localhost:8080'
    
    try:
        llm = get_llm_model('localai', model_name='gpt-3.5-turbo')
        print("✅ LLM model created successfully")
        print(f"📝 Model type: {type(llm)}")
        
        # Test a simple completion (only if LocalAI is running)
        client = get_localai_client()
        if client and client.is_available():
            try:
                response = llm.invoke("Hello, how are you?")
                print(f"🤖 Test response: {response.content[:100]}...")
            except Exception as e:
                print(f"⚠️ Could not test completion: {e}")
        
    except Exception as e:
        print(f"❌ Error creating LLM model: {e}")

def main():
    """Main test function"""
    print("🧪 Testing Free Job Application Agent - LocalAI Integration")
    print("=" * 60)
    
    test_localai_client()
    test_llm_provider()
    
    print("\n" + "=" * 60)
    print("✨ Test completed!")
    print("\n📝 Notes:")
    print("- If LocalAI server is not running, some tests will fail")
    print("- To start LocalAI: docker run -p 8080:8080 quay.io/go-skynet/local-ai:latest")
    print("- The agent will fall back to other providers if LocalAI is unavailable")

if __name__ == "__main__":
    main()

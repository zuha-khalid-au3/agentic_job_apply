"""
LocalAI Client Utility
Provides a wrapper for interacting with LocalAI as a drop-in replacement for OpenAI
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class LocalAIClient:
    """
    A client for interacting with LocalAI API
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = "not-needed"):
        """
        Initialize LocalAI client
        
        Args:
            base_url: LocalAI server URL
            api_key: API key (not needed for LocalAI but required by OpenAI client)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key
        )
        
    def is_available(self) -> bool:
        """
        Check if LocalAI server is available
        
        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LocalAI server not available: {e}")
            return False
    
    def list_models(self) -> list:
        """
        List available models in LocalAI
        
        Returns:
            list: List of available models
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def chat_completion(self, messages: list, model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using LocalAI
        
        Args:
            messages: List of messages
            model: Model name to use
            **kwargs: Additional parameters
            
        Returns:
            Dict: Chat completion response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise


def get_localai_client() -> Optional[LocalAIClient]:
    """
    Get LocalAI client if available
    
    Returns:
        LocalAIClient or None if not available
    """
    localai_url = os.getenv("LOCALAI_URL", "http://localhost:8080")
    client = LocalAIClient(base_url=localai_url)
    
    if client.is_available():
        logger.info(f"LocalAI server available at {localai_url}")
        return client
    else:
        logger.warning("LocalAI server not available")
        return None


def create_localai_llm(model_name: str = "gpt-3.5-turbo"):
    """
    Create a LangChain-compatible LLM using LocalAI
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        ChatOpenAI instance configured for LocalAI
    """
    from langchain_openai import ChatOpenAI
    
    localai_url = os.getenv("LOCALAI_URL", "http://localhost:8080")
    
    return ChatOpenAI(
        base_url=f"{localai_url}/v1",
        api_key="not-needed",
        model=model_name,
        temperature=0.7
    )


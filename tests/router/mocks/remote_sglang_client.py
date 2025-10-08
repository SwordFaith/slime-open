"""
Remote SGLang client for E2E testing.
"""

import httpx
from typing import Optional


class RemoteSGLangClient:
    """Client to interact with remote SGLang server via HTTP."""

    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)

    def health_check(self) -> bool:
        """Check if remote SGLang is available."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 100) -> Optional[dict]:
        """Generate text via remote SGLang."""
        if not self.health_check():
            return None

        payload = {"text": prompt, "sampling_params": {"max_new_tokens": max_tokens}}
        response = self.client.post(f"{self.base_url}/generate", json=payload)
        return response.json()

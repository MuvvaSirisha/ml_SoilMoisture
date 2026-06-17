import requests
import json
import Config

class QueryGuard:
    """
    Guardrail component to prevent prompt injections and off-topic queries.
    Uses the local Ollama LLM to classify if a query is within the domain scope.
    """

    def __init__(self):
        self.enabled = getattr(Config, 'GUARDRAILS_ENABLED', False)
        self.url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = getattr(Config, 'OLLAMA_TIMEOUT', 60)
        self.prompt_template = getattr(Config, 'GUARDRAILS_PROMPT', "")

    def is_safe(self, query: str) -> dict:
        """
        Validates if the query is safe and on-topic.
        
        Returns:
            dict: {
                "safe": bool,
                "reason": str
            }
        """
        if not self.enabled:
            return {"safe": True, "reason": ""}
            
        # Basic keyword bypass for simple queries to save time
        fast_keywords = ["mean", "minimum", "maximum", "slope", "trend", "help", "exit", "quit"]
        if len(query.split()) < 5 and any(k in query.lower() for k in fast_keywords):
            return {"safe": True, "reason": ""}

        prompt = self.prompt_template.format(query=query)

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.1
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            raw_text = response.json().get('response', '{}')
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            
            parsed = json.loads(raw_text)
            
            # Default to safe if the model fails to return a clear boolean
            is_safe = parsed.get("safe", True)
            reason = parsed.get("reason", "Out of domain or unsafe query.") if not is_safe else ""
            
            return {"safe": is_safe, "reason": reason}

        except Exception as e:
            # If Ollama fails or times out, we default to safe to not block the user, 
            # or you can default to False depending on strictness. Here we default to True.
            print(f"⚠️ Guardrails check failed ({e}). Proceeding normally.")
            return {"safe": True, "reason": ""}

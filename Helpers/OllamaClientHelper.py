import requests
import base64
import os

class OllamaClientHelper:
    def __init__(self, base_url="http://localhost:11434", model="llama3.2-vision"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """Reads and base64 encodes an image file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat(self, prompt: str, image_path: str = None) -> str:
        """
        Sends a prompt (optionally with an image) to the Ollama model.

        :param prompt: The user's question or instruction
        :param image_path: Optional path to a local image file
        :return: The model's response text
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        if image_path:
            encoded_image = self._encode_image(image_path)
            payload["messages"][0]["images"] = [encoded_image]

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")


# usage
# response = ollama.chat(
#     "Describe what’s in this image and check if it contains a cowboy hat.",
#     image_path="images/hat.jpg"
# )

from groq import Groq

from app.core.config import settings


class LLMClient:
    def __init__(self):
        self.client = Groq(
            api_key=settings.groq_api_key
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
        }

        if json_mode:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        response = (
            self.client.chat.completions.create(
                **kwargs
            )
        )

        return response.choices[0].message.content

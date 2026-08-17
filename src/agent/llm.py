import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class FraudLLM:

    def __init__(self):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = (
            "openai/gpt-oss-120b"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        response = (
            self.client
            .chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )
        return content if content is not None else ""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def llm_refine(prediction: dict) -> dict:
    prompt = f"""
Refine this prediction using manufacturing constraints.
Return JSON only.

Input:
{prediction}
"""

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL"),
        input=prompt
    )

    return response.output_parsed

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def llm_refine(prediction: dict) -> dict:
    prompt = f"""
Refine this prediction using the most abstract mathematics to exactly determine the seven positions with assertive separation.
Return JSON only.

Input:
{json.dumps(prediction)}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content
    if result:
        return json.loads(result)
    else:
        return prediction

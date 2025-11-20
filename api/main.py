from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# classes
class GenerateRequest(BaseModel):
    """
    What client sends when asking the LLM to generate steps.
    """
    play_id: str
    goal: str
    roles: List[str]


class StepDraft(BaseModel):
    """
    Maps directly onto the createPlayStep GraphQL input,
    excluding DB-managed fields (id, play_id, client_id).
    """
    step_name: str
    step_description: Optional[str] = None
    step_num: int
    step_role_name: Optional[str] = None


class GenerateStepsResponse(BaseModel):
    """
    What /generate endpoint returns to the frontend.
    """
    steps: List[StepDraft]


# call llm
def call_llm_for_steps(req: GenerateRequest) -> dict:
    roles_str = ", ".join(req.roles) if req.roles else ""

    system_prompt = f"""
You are given natural language and generate clear actionable steps in JSON format.

The schema is:

{{
  "steps": [
    {{
      "step_name": "string",
      "step_description": "string or null",
      "step_num": number,
      "step_role_name": "string or null"
    }}
  ]
}}

Schema information:
- "step_name" is the name of the step.
- "step_description" is a clear and concise explanation of the step.
- "step_num" must:
    - start at 1
    - be an integer
    - have no duplicates
    - be sequential with no gaps (1, 2, 3, ...)
- "step_role_name" is a job role that can most closely be assigned to complete the step.
    - You may ONLY choose from this list of roles:
      [{roles_str}]
    - If no role fits well, use null.

Rules:
- Return ONLY valid JSON.
- No explanations, no prose.
- step_num starts at 1 and increments by 1.
""".strip()

    user_prompt = f"""
Turn this into actionable steps:

{req.goal}

Return only JSON.
""".strip()

    resp = client.responses.create(
        model="gpt-5-mini-2025-08-07", 
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = resp.output_text

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM JSON: {e}. Raw text: {text!r}")


# Fastapi / CORS config

app = FastAPI()

origins = [
    "http://localhost:3000",
    "with-supabase-app-neon.vercel.app"
]


# Not very protective but this is just an assessment demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", response_model=GenerateStepsResponse)
async def generate(req: GenerateRequest) -> GenerateStepsResponse:
    try:
        llm_json = call_llm_for_steps(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Validate & normalize llm response with Pydantic
    try:
        response = GenerateStepsResponse(**llm_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid LLM JSON: {e}")

    return response

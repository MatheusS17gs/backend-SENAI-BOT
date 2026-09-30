import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, BadRequestError
import logging
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Optionally create an OpenAI client at startup if available
client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

'''
JSON
{
 "mensagem": "<qualquer texto>"
}
'''
class ChatMessage(BaseModel):
    mensagem: str
    id_anterior: str | None = None

@app.post("/chat")
async def chat(message: ChatMessage):
    if not message.mensagem or not message.mensagem.strip():
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    global client
    if client is None:
        client = OpenAI(api_key=api_key)

    try:
        payload = {
            "model": "gpt-4o-mini",
            "input": message.mensagem,
            "max_output_tokens": 500,
            "temperature": 0.3,
            "instructions": "Você é um assistente útil e amigável chamado Jarvis. Responda de forma clara e concisa, fornecendo informações relevantes e úteis. Evite respostas vagas ou genéricas. Seja educado e profissional em suas respostas. E você deve responder apenas conteúdos relacionados ao SENAI.",
        }

        if message.id_anterior:
            payload["previous_response_id"] = message.id_anterior

        resp = client.responses.create(**payload)

        reply = None
        try:
            reply = getattr(resp, "output_text", None)
        except Exception:
            reply = None

        if not reply:
            try:
                reply = resp.output[0].content[0].text
            except Exception:
                try:
                    reply = resp["output"][0]["content"][0]["text"]
                except Exception:
                    reply = None

        if not reply:
            reply = str(resp)

        reply = (reply or "").strip()
    except AuthenticationError as e:
        logging.exception("OpenAI authentication failed")
        raise HTTPException(
            status_code=401,
            detail="Chave da OpenAI inválida, expirada ou bloqueada. Gere uma nova chave no painel da OpenAI.",
        ) from e
    except BadRequestError as e:
        logging.exception("OpenAI bad request")
        raise HTTPException(status_code=400, detail=f"Erro de requisição para a OpenAI: {str(e)}") from e
    except Exception as e:
        tb = traceback.format_exc()
        logging.exception("Error calling OpenAI Responses API")
        last_line = tb.splitlines()[-1] if tb else ""
        detail = f"{type(e).__name__}: {str(e)} -- {last_line}"
        raise HTTPException(status_code=500, detail=detail)

    return {"mensagem": message.mensagem, "resposta": reply, "id": resp.id}
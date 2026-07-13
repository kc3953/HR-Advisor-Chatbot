import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 1. Setup CORS and Static Files (Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the frontend at the root URL
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 2. Vertex AI Client Setup
# In Cloud Run, authentication is automatic via the service account.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set")
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

logger.info(f"Initializing Vertex AI client with project: {PROJECT_ID}, location: {LOCATION}")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 3. Define the System Prompt
# Prompt organization method: RISEN (Role, Instructions, Steps, End Goal, Narrowing)
SYSTEM_INSTRUCTION = """
## ROLE
You are a senior HR Advisor specializing in People Analytics. You assist employees and HR
professionals with questions about HR policies, workplace guidelines, employee benefits,
performance management, and people analytics concepts.

## INSTRUCTIONS
When answering a question, follow these steps in order:
1. Identify whether the question is in-domain (HR/people analytics), out-of-scope, or a safety situation.
2. If in-domain: provide a structured, bulleted answer with bold headers.
3. If out-of-scope: trigger the escape hatch response exactly as specified below.
4. If a safety or distress signal is detected: trigger the safety protocol immediately before anything else.
5. If legal advice is requested: redirect to legal counsel, then optionally provide general context only.

## END GOAL
Help users understand HR concepts, policies, and people analytics clearly and safely —
while protecting them from harm and keeping all answers within the HR domain.

## NARROWING (Positive Constraints)
- Only answer questions related to HR, people analytics, workplace policies, employee benefits,
  performance management, talent acquisition, workforce planning, and employee relations.
- Provide answers in a concise, bulleted format with bold headers.
- When citing policies or data, be clear about what is general best practice vs. company-specific.
- Always maintain a professional, empathetic, and neutral tone.

SAFETY PROTOCOL — DISTRESSED USERS:
- If a user expresses distress, mentions harassment, discrimination, mental health struggles,
  thoughts of self-harm, or any crisis situation, immediately respond with empathy and provide
  the following resources:
    * Employee Assistance Program (EAP): contact your HR department for your company's EAP number
    * Crisis Text Line: Text HOME to 741741
    * National Suicide Prevention Lifeline: 988
  Then advise them to speak directly with a human HR representative or manager.
- Never attempt to resolve legal complaints, harassment claims, or mental health crises yourself.
  Always escalate these to a qualified human professional.

SCOPE BOUNDARIES (Out-of-Scope Categories):
The following categories are outside your scope. Respond to each with the exact phrase shown:

1. TECHNOLOGY & CODING — questions about programming, software, IT systems, or technical tools
   that are unrelated to HR software (e.g. "write me a script", "how does TCP/IP work"):
   → "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."

2. PERSONAL LIFESTYLE — questions about food, travel, entertainment, sports, hobbies, or general life advice
   (e.g. "best restaurants in NYC", "who won the game"):
   → "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."

3. FINANCIAL & LEGAL ADVICE — questions requiring specific legal rulings, investment advice, tax guidance,
   or any professional opinion outside HR scope (e.g. "should I buy this stock", "how do I file my taxes"):
   → "This question may require legal or financial expertise. Please consult your company's legal counsel, an employment attorney, or a licensed financial advisor."

EXAMPLES (follow this format and tone exactly):

Example 1 — In-domain, people analytics concept:
User: What is employee attrition rate and how is it calculated?
Assistant:
- **Definition:** Employee attrition rate measures the percentage of employees who leave an organization over a given period.
- **Formula:** (Number of employees who left ÷ Average headcount) × 100
- **Example:** If 10 employees left out of an average of 200, the attrition rate is 5%.
- **Note:** This is a general best practice formula; your company may track this differently in its HRIS.

Example 2 — Out-of-scope, escape hatch:
User: Can you help me write a Python script?
Assistant: I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource.

Example 3 — Safety protocol, distressed user:
User: My manager has been bullying me for months and I feel like I can't take it anymore.
Assistant: I'm really sorry to hear you're going through this — what you're feeling is valid, and you deserve support. Please reach out to the following resources right away:
- **Your HR department** — report the situation to a human HR representative who can take formal action.
- **Employee Assistance Program (EAP)** — contact your HR team for your company's confidential EAP number.
- **Crisis Text Line** — Text HOME to 741741 for free, confidential support 24/7.
- **National Suicide Prevention Lifeline** — call or text 988 if you are in crisis.
You do not have to handle this alone. Please speak with a qualified HR professional or counselor directly.

Example 4 — Uncertainty / legal boundary:
User: Can I sue my company for wrongful termination?
Assistant: This question may require legal expertise. Please consult your company's legal counsel or an employment attorney. What I can share is general context:
- **Wrongful termination** typically refers to being fired in violation of employment law or a contract.
- Common grounds include discrimination, retaliation, or breach of contract.
- An employment attorney can assess whether your specific situation warrants legal action.
"""

# 4. Python Backstop — post-generation safety classifier
# Catches cases where the LLM fails to apply the system prompt correctly.

DISTRESS_KEYWORDS = [
    "harass", "bully", "bullying", "assault", "abuse", "discriminat",
    "hopeless", "worthless", "don't want to be here", "can't take it",
    "end it all", "hurt myself", "self-harm", "suicid", "overwhelmed and",
    "breaking down", "mental breakdown",
]

OUT_OF_SCOPE_KEYWORDS = [
    "recipe", "cook", "restaurant", "netflix", "movie", "song", "sport",
    "football", "basketball", "nfl", "nba", "super bowl", "tourist",
    "vacation", "travel", "stock market", "crypto", "bitcoin", "investment",
    "write a script", "python script", "javascript", "html code", "hack",
    "write code", "debug", "tcp/ip", "machine learning model",
]

DISTRESS_FALLBACK = """I'm really sorry to hear you're going through a difficult time. What you're feeling matters, and you deserve support.

Please reach out to the following resources right away:
- **Your HR department** — speak with a human HR representative who can take formal action.
- **Employee Assistance Program (EAP)** — contact your HR team for your company's confidential EAP number.
- **Crisis Text Line** — Text HOME to 741741 for free, confidential support 24/7.
- **National Suicide Prevention Lifeline** — call or text **988** if you are in crisis.

You do not have to handle this alone. Please speak with a qualified HR professional or counselor directly."""

OUT_OF_SCOPE_FALLBACK = "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."


def backstop_classifier(user_message: str, llm_response: str) -> str | None:
    """
    Post-generation backstop. Returns an override response if the LLM
    failed to handle a distress signal or out-of-scope question correctly.
    Returns None if the LLM response looks correct.
    """
    msg_lower = user_message.lower()

    # Check 1: Distress signal in user message — ensure crisis resources are present
    if any(kw in msg_lower for kw in DISTRESS_KEYWORDS):
        if not any(marker in llm_response for marker in ["988", "741741", "EAP"]):
            logger.warning("Backstop triggered: distress signal detected, LLM missed safety protocol")
            return DISTRESS_FALLBACK

    # Check 2: Out-of-scope topic — ensure LLM refused correctly
    if any(kw in msg_lower for kw in OUT_OF_SCOPE_KEYWORDS):
        refusal_markers = ["only answer questions", "people analytics", "appropriate resource"]
        if not any(marker in llm_response.lower() for marker in refusal_markers):
            logger.warning("Backstop triggered: out-of-scope topic detected, LLM did not refuse")
            return OUT_OF_SCOPE_FALLBACK

    return None


class ChatRequest(BaseModel):
    message: str

@app.get("/favicon.ico", status_code=204)
async def favicon():
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "domain-chatbot"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request.message[:100]}...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3, # Low temp for deterministic factual answers
            ),
            contents=request.message
        )
        llm_response = response.text

        # Run Python backstop to catch LLM misses
        override = backstop_classifier(request.message, llm_response)
        if override:
            return {"response": override}

        logger.info(f"Generated response successfully")
        return {"response": llm_response}
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Error")

@app.get("/")
async def root():
    return {"message": "Chatbot API is running. Go to /static/index.html to chat."}

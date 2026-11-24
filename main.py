import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field

# Load environment variables from a local .env file (such as GROQ_API_KEY)
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="Product API with Groq RAG")

# Create the Groq client using the API key from the environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Anchor file paths to this script's folder so running uvicorn from another
# directory still finds product_notes.txt reliably
BASE_DIR = Path(__file__).resolve().parent
NOTES_PATH = BASE_DIR / "product_notes.txt"

# Split product_notes.txt into non-empty lines so each note acts as a small chunk
NOTES_CHUNKS: list[str] = [
    line.strip()
    for line in NOTES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


# Request schema for creating a new product with basic field rules
class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    in_stock: bool = True


# Response schema for items stored in the catalog
class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float
    in_stock: bool


# Request body expected by POST /products/ask
class AskProductRequest(BaseModel):
    question: str = Field(..., min_length=1)


# Strict output schema that Groq must match when answering product questions
class ProductAssistantResponse(BaseModel):
    answer: str
    referenced_notes: list[str]
    needs_human_review: bool


# Sample in-memory product catalog for local testing
products: list[Product] = [
    Product(id=1, name="Wireless Mouse", category="electronics", price=29.99, in_stock=True),
    Product(id=2, name="Desk Lamp", category="home", price=44.50, in_stock=True),
    Product(id=3, name="Mechanical Keyboard", category="electronics", price=89.00, in_stock=False),
]


def tokenize(text: str) -> set[str]:
    # Convert text to lowercase alphanumeric words for simple keyword matching
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_relevant_notes(query: str, top_k: int = 2) -> list[str]:
    # Score each note chunk by how many query words overlap with the chunk
    query_tokens = tokenize(query)
    scored: list[tuple[int, str]] = []

    for chunk in NOTES_CHUNKS:
        overlap = len(query_tokens.intersection(tokenize(chunk)))
        if overlap > 0:
            scored.append((overlap, chunk))

    # Sort highest overlap first and return the top_k matching notes
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


@app.get("/health")
def health() -> dict[str, str]:
    # Quick endpoint to verify the server is alive
    return {"status": "ok"}


@app.get("/products", response_model=list[Product])
def list_products() -> list[Product]:
    # Return the current product list
    return products


@app.post("/products/ask", response_model=ProductAssistantResponse)
def ask_product_assistant(payload: AskProductRequest) -> ProductAssistantResponse:
    # Fail fast with a clear HTTP error if the API key is missing
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    # Step 1 (Retrieval): pull the most relevant lines from product_notes.txt
    retrieved_notes = retrieve_relevant_notes(payload.question)
    context_block = "\n".join(retrieved_notes) if retrieved_notes else "No matching notes found."

    # Format current catalog items so the model can reference prices and stock status
    catalog_summary = "\n".join(
        f"- ID {p.id}: {p.name} (${p.price}, category={p.category}, in_stock={p.in_stock})"
        for p in products
    )

    # Step 2 (Prompting): instruct the model to stick strictly to provided data
    prompt = (
        "Answer using ONLY the product catalog and retrieved notes below. "
        "If the answer is not supported by the text, set needs_human_review to true. "
        "Return referenced_notes as note IDs like [\"NOTE-1\"].\n\n"
        f"Catalog:\n{catalog_summary}\n\n"
        f"Retrieved Notes:\n{context_block}\n\n"
        f"Question: {payload.question}"
    )

    # Step 3 (Generation): request structured JSON matching ProductAssistantResponse
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "product_assistant_response",
                "schema": ProductAssistantResponse.model_json_schema(),
            },
        },
    )

    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="Empty response from Groq")

    # Step 4 (Validation): parse JSON and validate types with Pydantic
    try:
        data = json.loads(content)
        return ProductAssistantResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Invalid structured output: {exc}") from exc
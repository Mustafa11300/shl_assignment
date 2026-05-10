# SHL Conversational Assessment Recommender

A conversational agent that helps hiring managers discover the right SHL Individual Test Solutions through natural dialogue. Built as a stateless FastAPI service.

## Features

- **Clarify** vague queries before recommending
- **Recommend** 1-10 assessments with names, URLs, and test type codes
- **Refine** recommendations when constraints change mid-conversation
- **Compare** assessments using grounded catalog data
- **Refuse** off-topic, legal, and prompt injection attempts
- **Validate** every URL against the 377-item SHL catalog

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export GEMINI_API_KEY=your_key_here

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a Java developer"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Architecture

1. **Query extraction** from conversation history
2. **Hybrid retrieval** via FAISS (semantic) + keyword boosting
3. **LLM generation** (Gemini 2.5 Flash) with catalog context
4. **Response validation** — every URL checked against catalog

## Evaluation

```bash
# Run against sample conversations
python eval/evaluate.py --url http://localhost:8000 --traces ~/Downloads/GenAI_SampleConversations
```

## Tech Stack

- **LLM**: Gemini 2.5 Flash (free tier)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (in-memory)
- **Framework**: FastAPI + uvicorn
- **Deployment**: Render (Docker)

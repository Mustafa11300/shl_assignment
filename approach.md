# Approach Document — SHL Conversational Assessment Recommender

## Architecture

The system is a stateless FastAPI service with two endpoints (`GET /health`, `POST /chat`). Each `/chat` call processes the full conversation history through a three-stage pipeline:

1. **Query Extraction** — User messages are combined into a semantic search query, weighting recent turns more heavily. Named assessments (OPQ, Verify G+, etc.) mentioned in the conversation are detected via abbreviation mapping and direct catalog lookup.

2. **Hybrid Retrieval** — A FAISS index over 377 catalog items (embedded via `all-MiniLM-L6-v2`) performs cosine similarity search. Results are re-ranked with keyword boosting: exact technology/skill name matches in the query get a score bump, ensuring precise items surface alongside semantically similar ones.

3. **LLM Generation + Validation** — Retrieved items are injected into a structured system prompt and sent to Gemini 2.5 Flash. The model returns JSON matching the exact response schema. Post-processing validates every URL and name against the catalog, deduplicates, and caps at 10 recommendations.

## Retrieval Setup

The catalog (377 Individual Test Solutions from `dataset.txt`) is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Each item's embedding text combines: name, description, test type categories, job levels, languages, and duration. The FAISS `IndexFlatIP` index is pre-built and persisted to disk for sub-second cold starts.

Hybrid scoring: `final_score = cosine_similarity + keyword_boost`. Keywords are matched against assessment names with common stop words filtered. Mentioned assessments (detected by abbreviation or full name) are always included in the retrieval context, ensuring comparison and refinement requests have access to previously recommended items.

## Prompt Design

A single system prompt encodes six behavioral rules: clarify vague queries, recommend when enough context exists, refine without starting over, compare using catalog data only, refuse off-topic/legal questions, and pushback on potentially bad decisions. The prompt instructs JSON output with strict field definitions for `reply`, `recommendations`, and `end_of_conversation`.

Key prompt decisions: (a) temperature 0.3 for consistency, (b) catalog items provided in full (name, URL, description, duration, languages, test type) so the LLM never hallucinates metadata, (c) explicit rules for when to recommend vs clarify (specific role + skills = recommend; "I need an assessment" = clarify).

## Evaluation Approach

Evaluation uses the 10 public conversation traces (C1–C10). Each trace is parsed to extract user turns and expected final recommendations. The replay harness sends user messages sequentially against the live API, collecting the agent's final recommendation set.

**Recall@10** is computed per trace: `|intersection(predicted, expected)| / |expected|`. Behavior probes test: schema compliance, URL validity, off-topic refusal, vague-query clarification, and mid-conversation edit handling.

## What Didn't Work

- **Gemini 2.0 Flash** hit rate limits on the free tier; switched to Gemini 2.5 Flash which had available quota.
- **2048 max_output_tokens** caused truncated JSON for longer recommendation lists; increased to 4096.
- Early prompt versions would sometimes recommend on vague queries. Adding explicit examples of "vague" vs "specific" queries in the prompt fixed this.

## AI Tools Used

Agentic coding assistant (Gemini-based) was used for scaffolding the project structure, writing boilerplate (Pydantic models, Dockerfile), and iterating on prompt engineering. All design decisions, retrieval strategy, and evaluation logic were manually designed and validated.

## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | Gemini 2.5 Flash (free tier) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (in-memory, 377 vectors) |
| Framework | FastAPI + uvicorn |
| Deployment | Render (Docker, free tier) |

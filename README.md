# AI Smart Travel Planning Assistant

**Agentic AI + Agentic RAG** - a notebook-based travel assistant that answers questions from travel-guide PDFs, plans itineraries, compares destinations, estimates budgets and gives weather advisories.

Developed for the BCS-752 Mini Project / Internship Assessment at Ajay Kumar Garg Engineering College (AKGEC), Ghaziabad.
Author: Pranjal Sharma

## What it does

- Loads travel-guide PDFs and splits the text into overlapping chunks (700 characters, 120 overlap).
- Indexes the chunks with TF-IDF and retrieves the top-k passages by cosine similarity.
- Answers questions using only the retrieved text and names the source documents.
- Offers five capabilities: question answering, itinerary, destination comparison, budget estimate and weather advisory.
- Implements two agentic designs on the same retrieval layer and tools:
  - **RouterAgent** - classifies a request into one of five intents and hands it to a specialised agent.
  - **SingleAgent** - one Claude agent that decides when to call three tools (`get_weather`, `estimate_budget`, `search_knowledge_base`) in a `tool_use` loop (max 4 turns).
- Runs **without an API key** using a built-in mock LLM, or with the real Claude API when `ANTHROPIC_API_KEY` is set.

## Architecture

```
PDFs -> pypdf text extraction -> chunking (700/120) -> TF-IDF vector store
                                                           |
User query -> RouterAgent or SingleAgent -> agents / tools -> llm_call() -> answer + sources
                                                            (Claude API, or mock LLM)
```

## How to run

1. Install Python 3 and Jupyter.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Use the real Claude model by setting your key *before* starting Jupyter:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key-here"
   export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"   # optional override
   ```
   Never commit your key to the repository.
4. Open `Agentic_AI_Travel_Assistant.ipynb` and run all cells. Two sample PDFs (Goa and Manali) are generated in `sample_travel_guides/`.
5. Ask something with the helper, for example:
   ```python
   chat_once("What is the best time to visit Goa?")
   chat_once("How much would a 4 day budget trip to Goa cost?", destination="Goa", days=4, style="budget")
   ```

## Saved results (from the committed run)

The outputs saved in the notebook were produced in **mock-LLM mode** (`MOCK_MODE = True`, no API key set), so they verify retrieval, tools and control flow, not the quality of generated text.

| Check | Observed |
|---|---|
| Sample PDFs loaded | 2 (943 and 946 characters) |
| Chunks indexed | 4 |
| Search "best time to visit for snow" | Manali 0.261, Goa 0.123 |
| Budget, 4-day budget trip to Goa | 120 USD |
| Intents routed as expected | 5 of 5 |

## Limitations

- The sample corpus is two short synthetic guides, so source attribution is imprecise (a Goa question can also cite the Manali guide).
- Retrieval is TF-IDF (word matching), not dense embeddings.
- The router does not extract destination, days or city from the sentence; they are passed as keyword arguments.
- The SingleAgent executes one tool call per turn.
- Weather data is simulated for three cities, and budgets come from a fixed per-day cost model in USD.
- Answer quality with a live Claude model has not been evaluated.
- There is no web interface; the notebook is the interface.

## Future work

Dense embeddings with a vector database (FAISS or Chroma), relevance thresholds and destination filtering, structured extraction of trip details by the LLM, parallel tool calls, a live weather API, an evaluation set with real Claude responses, and a web front-end.

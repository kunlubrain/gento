# gento

A lightweight LLM client.

## Features

- **Unified Client Interface**: Interact with multiple LLM providers (`Gemini`, `OpenAI`, `Volcengine/Ark`) via a single `LLMClient`.
- **Default Model Support**: Defaults to `google/gemini-3.5-flash` if no model is specified during initialization, or set a default model for subsequent calls.
- **Environment-based Credentials**: No need to hardcode API keys in the client — seamlessly picks up `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `VOLC_API_KEY` / `ARK_API_KEY`.
- **Structured Pydantic Outputs**: Validate and parse LLM outputs into Pydantic models with automatic retries on schema parse failures.
- **Tenacity Retry Logic**: Automatic retry with exponential backoff and jitter for transient errors and schema validation failures.
- **Rate Limiting**: Built-in rate limiting using `aiolimiter`.
- **Web Search**: Grounding support with `enable_search=True`.

## Installation

```bash
pip install gento
```

or with Poetry:

```bash
poetry add gento
```

## Quick Start

### Basic Text Generation

```python
import asyncio
from gento import LLMClient

async def main():
    # Defaults to model="google/gemini-3.5-flash"
    client = LLMClient()

    # Generate response using default model
    response = await client.generate("Explain quantum computing in one sentence.")
    print("Content:", response.content)

    # Override model for specific call
    response_openai = await client.generate(
        "Explain special relativity in one sentence.",
        model="openai/gpt-4o"
    )
    print("OpenAI Content:", response_openai.content)

asyncio.run(main())
```

### Structured Output with Pydantic

```python
import asyncio
from pydantic import BaseModel, Field
from gento import LLMClient

class MovieInfo(BaseModel):
    title: str = Field(description="Title of the movie")
    release_year: int = Field(description="Release year")
    director: str = Field(description="Director of the movie")

async def main():
    client = LLMClient(model="google/gemini-3.5-flash")

    # Generate structured Pydantic model output
    movie: MovieInfo = await client.generate_structured(
        "Provide details about the movie Inception.",
        response_schema=MovieInfo,
    )

    print(f"Title: {movie.title}, Director: {movie.director}, Year: {movie.release_year}")

asyncio.run(main())
```

### Web Search Grounding

```python
import asyncio
from gento import LLMClient

async def main():
    client = LLMClient(model="google/gemini-3.5-flash")

    response = await client.generate(
        "What are the latest developments in AI technology this week?",
        enable_search=True,
    )
    print(response.content)

asyncio.run(main())
```

## Supported Models & Provider Resolution

```
google/gemini-3.5-flash
google/gemini-3.6-flash
google/gemini-3.7-flash
openai/gpt-4o
openai/gpt-4o-mini
openai/gpt-5
openai/gpt-4o-mini
openai/gpt-4o
volcengine/doubao-1.5-pro-32k
volcengine/doubao-pro-32k
```

## Environment Variables

- **Gemini**: `GEMINI_API_KEY`
- **OpenAI**: `OPENAI_API_KEY` (Optional: `OPENAI_BASE_URL`)
- **Volcengine/Ark**: `VOLCENGINE_API_KEY` (Optional: `VOLC_BASE_URL`)

## License

MIT

## TODO

Do not retry if API error:
File "/Users/kunlu/proj/gento/gento/adapters/gemini.py", line 119, in generate
raise APIError(f"Gemini API error: {str(e)}") from e

File "/Users/kunlu/proj/gento/gento/adapters/gemini.py", line 117, in generate
raise APIError(f"Gemini API error: {str(e)}") from e
gento.exceptions.APIError: Gemini API error: Object of type ModelMetaclass is not JSON serializable

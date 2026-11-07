# Vehicle Manual Assistance

A Retrieval-Augmented Generation (RAG) assistant that answers questions about a vehicle owner's manual (currently the **Maruti Suzuki Fronx** manual). Ask a question in plain English and get an answer grounded in the manual, along with citations of the content it used.

## How it works

1. **Embed**: manual content is converted into vector embeddings with `Qwen/Qwen3-Embedding-4B` (via `sentence-transformers`).
2. **Retrieve**: your question is embedded the same way, and the most relevant chunks of the manual are fetched.
3. **Generate**: the retrieved context and your question are passed to `meta-llama/Llama-3.2-3B-Instruct`, which writes the answer and cites the source content.
4. **Evaluate**: `eval_rag.py` measures how well the pipeline performs.

## Project structure

```
Vehicle_Manual_Assistance/
├── main.py            # Entry point: runs the RAG pipeline
├── models.py          # Loads the generator and embedding models, builds the prompt
├── eval_rag.py        # Evaluation script for the RAG pipeline
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

## Models used

| Role            | Model                              |
| --------------- | ---------------------------------- |
| Generator (LLM) | `meta-llama/Llama-3.2-3B-Instruct` |
| Embeddings      | `Qwen/Qwen3-Embedding-4B`          |

The generator is loaded in `float16` with `device_map="auto"`, so it uses a GPU when one is available and falls back to CPU otherwise. CPU inference works but is slow.

## Requirements

- Python 3.9+
- A Hugging Face account and access token
- Approved access to the gated Llama model: request it on the [Llama-3.2-3B-Instruct page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- A CUDA-capable GPU is recommended

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/williamtom3010/Vehicle_Manual_Assistance.git
cd Vehicle_Manual_Assistance
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Hugging Face token**

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

> Never commit `.env` or hardcode your token in the code. `.env` is already listed in `.gitignore`.

## Usage

Run the assistant:

```bash
python main.py
```

Run the evaluation:

```bash
python eval_rag.py
```

## Example

```
Question: How do I turn on the rear defogger?
Answer:   Press the rear defogger switch on the climate control panel... (Source: Climate Control section)
```

_Replace this with a real example from your own runs._

## Configuration

Generation settings live in `models.py` inside `gen_ans()`:

- `max_new_tokens`: maximum length of the answer (default `500`)
- `temperature`, `top_k`, `top_p`: sampling settings (these only take effect when `do_sample=True`)

To use a different manual or model, change the model name in `load_generator_model()` / `load_embedding_model()` and update the prompt text in `gen_ans()`.

## Troubleshooting

- **`401` / gated repo error**: make sure your token is in `.env` and that your Hugging Face account has been granted access to the Llama model.
- **Out of memory**: the embedding model (4B) and generator (3B) are both large. Close other GPU programs, or switch to a smaller embedding model such as `Qwen/Qwen3-Embedding-0.6B`.
- **Slow responses**: expected on CPU. A GPU makes a large difference.

## Roadmap

- [ ] Support multiple vehicle manuals
- [ ] Add a web UI (Streamlit or Gradio)
- [ ] Show page numbers in citations
- [ ] Add more evaluation metrics

## License

Add a license of your choice (for example MIT) and note it here.

## Acknowledgements

- [Meta Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) and [Sentence-Transformers](https://www.sbert.net/)

# prompt the generation model
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load model directly
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

def load_generator_model():
    
    model_path ="meta-llama/Llama-3.2-3B-Instruct"
    generator_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        device_map="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # model = model.to(device)
    print("Model has been loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer has been loaded")

    return generator_model,tokenizer


generator_model,tokenizer = load_generator_model()


def gen_ans(query,context):

    
    generator_model.eval()
    
    prompt = f"""You are an expert car enthusiast guide.
    Use the following context from the Maruti suzuki fronx manual to answer the question.
    Also provide citation of the content that you generate.
    Context:
    {context}

    Question:
    {query}

    Answer:"""

    # llm generator
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(generator_model.device)


    output = generator_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                top_k=30,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return answer


def load_embedding_model():
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    print("Embedding dim:", embedding_model.get_sentence_embedding_dimension())
    return embedding_model
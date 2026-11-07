# evaluation metrics 
# Retrieval evaluation metrics
import re
import pandas as pd
import chromadb 
import models
from rouge_score import rouge_scorer  
import sacrebleu  


benchmark_dataset_path = "data/Fronx_2024_RAG_Evaluation_Dataset_50.csv"
INDEX_DIR  = "fronx_manual_vector_db"

# choose embedding model
embedding_model = models.load_embedding_model()

# call vector db collection
client = chromadb.PersistentClient(path=INDEX_DIR)
collection = client.get_or_create_collection(
    name="fronx_manual",
    metadata={"hnsw:space": "cosine"},
)

# load eval dataset
df = pd.read_csv(benchmark_dataset_path)
df = df.dropna(subset=["question", "expected_context"]).reset_index(drop=True)
print(df.head())


def retrival_eval(K):   

    def words(text):
        return set(re.findall(r"[a-z0-9]+", str(text).lower()))

    def is_relevant(chunk, expected_context):
        gold = words(expected_context)
        overlap = len(gold & words(chunk)) / len(gold)
        return overlap >= 0.5      # 50% of gold words found in chunk

    precision_scores = []
    recall_scores = []
    hit_scores = []

    for _, row in df.iterrows():
        q_emb = embedding_model.encode(
            [row["question"]], prompt_name="query", normalize_embeddings=True
        )

        res = collection.query(query_embeddings=q_emb.tolist(), n_results=K)
        chunks = res["documents"][0]

        retrieved_words = set()
        flags = []
        relevant_count = 0

        
        for c in chunks:
                # recall
                retrieved_words |= words(c)
                # precision            
                rel  = is_relevant(c, row["expected_context"])
                relevant_count += rel
                flags.append(rel)

        precision_scores.append(relevant_count / K)

        # recall
        gold = words(row["expected_context"])
        recall = len(gold & retrieved_words) / len(gold)
        recall_scores.append(recall)    

        hit_scores.append(1 if any(flags) else 0)

    print(f"Questions evaluated: {len(recall_scores)}")
    print(f"Precision@{K}: {sum(precision_scores) / len(precision_scores):.3f}")
    print(f"Recall@{K}: {sum(recall_scores) / len(recall_scores):.3f}")
    print(f"Hit@{K}:       {sum(hit_scores) / len(hit_scores):.3f}")




# evaluation metrics 
# Generation evaluation metrics
# choose generation model

def gen_eval():
    def generate_answer(question, k=4):
        # retrieve
        q_emb = embedding_model.encode([question], prompt_name="query", normalize_embeddings=True)
        res = collection.query(query_embeddings=q_emb.tolist(), n_results=k)
        docs = res["documents"][0]

        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))

        answer = models.gen_ans(question,context)
        return answer, docs




    df = df.dropna(subset=["question", "expected_answer"]).reset_index(drop=True)

    answers, contexts = [], []
    for q in df["question"]:
        a, ctx = generate_answer(q)
        answers.append(a)
        contexts.append(ctx)

    df["generated_answer"] = answers
    df["contexts"] = contexts
    df.to_csv("generation_outputs.csv", index=False) 


    # ===========================================


    # rouge and blue

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_scores, r1_scores, rl_scores, r2_scores = [], [], []
    for ref, hyp in zip(df["expected_answer"], df["generated_answer"]):
        bleu_scores.append(sacrebleu.sentence_bleu(hyp, [ref]).score / 100)   # 0 to 1
        r = scorer.score(ref, hyp)            # order: (reference, generated)
        r1_scores.append(r["rouge1"].fmeasure)
        r2_scores.append(r["rouge2"].fmeasure)
        rl_scores.append(r["rougeL"].fmeasure)

    df["bleu"], df["rouge1"], df["rougeL"] = bleu_scores, r1_scores, rl_scores
    print(df[["bleu", "rouge1", "rougeL"]].mean().round(3))

    # Answer Relevance and Answer Correctness

    def emb(texts):
        return embedding_model.encode(texts, normalize_embeddings=True)

    ans_e = emb(df["generated_answer"].tolist())
    ref_e = emb(df["expected_answer"].tolist())
    q_e = embedding_model.encode(df["question"].tolist(), prompt_name="query", normalize_embeddings=True)

    # correctness: how close is the generated answer to the expected answer?
    df["answer_correctness"] = (ans_e * ref_e).sum(axis=1)

    # relevance: how close is the generated answer to the question?
    df["answer_relevance"] = (ans_e * q_e).sum(axis=1)

    print(df[["answer_correctness", "answer_relevance"]].mean().round(3))



K = 5
retrival_eval(K)
gen_eval()
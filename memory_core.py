# memory_core.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class MemoryCore:
    """
    Minimal RAG-style memory:
    - uses a sentence embedding model to embed text
    - stores (embedding, text) in a list
    - can retrieve top-k relevant memories for a new query
    You then prepend those memories to your generation prompt.
    This is the standard "persistent memory" trick used in modern RAG systems:
    external vector store feeding context back to the LLM at runtime,
    rather than editing weights every time. :contentReference[oaicite:21]{index=21}
    """

    def __init__(self, embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # any sentence embedding model; using a public MiniLM-like model is common today
        # for semantic retrieval. You can swap this with something local/quantized.
        self.tok = AutoTokenizer.from_pretrained(embed_model_name)
        self.enc = AutoModel.from_pretrained(embed_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc.to(self.device)
        self.mem = []  # list of (embedding_tensor_cpu, text)

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        """
        Get a single vector for text by mean-pooling last hidden states.
        This is the usual recipe for sentence embeddings in MiniLM/SBERT:
        mean-pool token embeddings, then normalize. (Same approach shown in a
        ton of retrieval/RAG tutorials in 2024–2025.) :contentReference[oaicite:22]{index=22}
        """
        batch = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        out = self.enc(**batch)
        # out.last_hidden_state: [1, seq_len, hidden]
        emb = out.last_hidden_state.mean(dim=1)  # [1, hidden]
        emb = F.normalize(emb, p=2, dim=1)       # cosine normalize
        return emb.squeeze(0).cpu()              # store on CPU

    def remember(self, text: str):
        vec = self._embed(text)
        self.mem.append((vec, text))

    def recall(self, query: str, k=3):
        if not self.mem:
            return []
        q = self._embed(query)  # [hidden]
        sims = []
        for i, (mem_vec, mem_text) in enumerate(self.mem):
            sim = torch.dot(q, mem_vec)  # cosine because both normalized
            sims.append((sim.item(), mem_text))
        sims.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in sims[:k]]

    def build_augmented_prompt(self, query: str):
        """
        Pull top memories and prepend them.
        You feed THIS to your tuned LLM to generate an answer.
        """
        top_mem = self.recall(query, k=3)
        memory_block = ""
        if top_mem:
            memory_block = "### MEMORY (persistent info):\n" + "\n".join(
                f"- {m}" for m in top_mem
            ) + "\n\n"
        prompt = memory_block + "### USER REQUEST:\n" + query + "\n\n### ASSISTANT:\n"
        return prompt

if __name__ == "__main__":
    core = MemoryCore()
    core.remember("User prefers concise answers about robot drivetrain tuning.")
    core.remember("User robot uses swerve drive with custom PID on elevator stage.")
    core.remember("Competition is Reefscape, they score coral and algae at L4.")

    q = "How should I tune my elevator PID so it doesn't overshoot L4 coral placement?"
    aug = core.build_augmented_prompt(q)
    print(aug)

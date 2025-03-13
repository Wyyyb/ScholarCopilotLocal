import faiss
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import glob
from tqdm import tqdm
from itertools import chain
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class FaissFlatSearcher:
    """FAISS-based dense retrieval index with inner product similarity."""

    def __init__(self, init_reps: np.ndarray):
        self.index = faiss.IndexFlatIP(init_reps.shape[1])

    def add(self, p_reps: np.ndarray):
        """Add new passage embeddings to the FAISS index."""
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        """Search for top-k nearest neighbors for given query embeddings."""
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool = False):
        """Perform batched nearest neighbor search."""
        num_query = q_reps.shape[0]
        all_scores, all_indices = [], []

        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)

        return np.concatenate(all_scores, axis=0), np.concatenate(all_indices, axis=0)


def pickle_load(path):
    """Load serialized data from a pickle file."""
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    """Save data to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_index_and_data(index_files):
    """Load passage representations and build FAISS index."""
    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))

    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    return retriever, look_up


def configure_faiss_for_gpu(retriever):
    """Optimize FAISS index for GPU usage if available."""
    num_gpus = faiss.get_num_gpus()

    if num_gpus == 0:
        print("No GPU found or using faiss-cpu. Running on CPU.")
    else:
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co, ngpu=num_gpus)

    return retriever


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token representation based on attention mask."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_query_embedding(model, tokenizer, query, max_length=4096):
    """Generate dense embedding for a given query using transformer model."""
    batch_dict = tokenizer([query], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


if __name__ == "__main__":
    # Load FAISS index and lookup table
    index_files = glob.glob("/data/yubowang/ScholarCopilotLocal/tevatron_retrieval/corpus*.pkl")  # Replace with actual path
    retriever, look_up = load_index_and_data(index_files)
    retriever = configure_faiss_for_gpu(retriever)

    # Load Transformer Model
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')

    # Define Query
    task = 'Given a paper passage, retrieve the most proper paper to cite next.'
    query = 'Transformers'
    query = get_detailed_instruct(task, query)

    query_embedding = get_query_embedding(model, tokenizer, query)

    # Perform Search
    k = 10
    scores, indices = retriever.search(query_embedding, k)
    documents = [look_up[i] for i in indices[0]]
    # Output Results
    print("Scores:", scores)
    print("Indices:", indices)
    print("Retrieved Documents:", [look_up[i] for i in indices[0]])


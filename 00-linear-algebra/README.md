# linear-algebra

> Linear algebra primitives implemented from scratch in pure Python, verified against NumPy.
 
## What this covers
 
 - Vectors & matrices
 - Subspaces
 - Orthogonality
 - Eigenvalues
 - SVD implementation
 - Matrix multiplication
 - Dot products

## What clicked
 
     Cosine similarity is where it clicked. While building a RAG pipeline, I had been calling a cosine similarity function to rank retrieved documents and treating it as a black box that returned a "relevance score." When I derived cos(θ) = (r·s) / (|r||s|) from the law of cosines, I realized I wasn't computing an abstract score. I was measuring the angle between two arrows in embedding space. The closer two documents are in meaning, the smaller that angle. I had been doing geometry for months without knowing it.
 
## Why it matters for what's ahead

     Every function here reappears in the models I'm about to build. `forward_pass` is a neural network and is literally a sequence of `mat_vec_multiply` calls. `attention_scores` is the Q·K dot product at the core of every transformer; when I get to attention, I will just be recognizing one I already wrote. `low_rank_approximation` previews LoRA (Phase 5), which uses the same low-rank property to make fine-tuning cheap. I'll understand the full mechanism when I get there. `l2_norm` is L2 regularization; the penalty term in loss functions that keeps weights small. None of these felt connected when I was reading about them separately. Building them from scratch in one file made the pattern more noticeable: ML is just applied linear algebra.

 
## Files
 
- `linear_algebra.py` — contains manually implemented functions
- `verify.py` — diffs each function in `linear_algebra.py` against its NumPy equivalent
- `linear-algebra-notes-01.pdf`, `linear-algebra-notes-02.pdf`  — full notes from the source material

## How to run
 
```bash
python linear_algebra.py
python verify.py
```
 
## Sources

 - "Introduction to Linear Algebra" w/ Gilbert Strang
 - MIT 18.06 
 - M4ML Crash Course Playlist

## Status
 
> Implementations done and verified. Synthesis written from memory.
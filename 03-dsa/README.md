# dsa

> Common algorithmic patterns implemented from scratch — the toolkit for classical ML problem-solving.

## What this covers

| Pattern | File | Status |
|---|---|---|
| Two Pointer | `two_pointer.py` | Done |
| Slow / Fast Pointer | `slow_fast_pointer.py` | Done |
| Sliding Window | `sliding_window.py` | Done |
| Binary Search | `binary_search.py` | Done |
| Top K Elements | `top_k_elements.py` | Done |
| Prefix Sum | `prefix_sum.py` | In Progress |
| Binary Tree Traversal | `binary_tree_traversal.py` | In Progress |
| Graph & Matrices | `graph_and_matrices.py` | In Progress |
| Backtracking | `backtracking.py` | In Progress |
| Dynamic Programming | `dynamic_programming.py` | Not Started |
| Monotonic Stack | `monotonic_stack.py` | Not Started |
| Overlapping Intervals | `overlapping_intervals.py` | Not Started |
| Bit Manipulation | `bit_manipulation.py` | Not Started |

## What clicked

Two pointers are the clearest example of replacing brute-force O(n²) with a greedy invariant. In `maxArea`, the argument is tight: the smaller boundary always moves inward because moving the taller one can only shrink width while still being capped by the same height. Once that clicked, `threeSum` followed naturally — sort, fix one index, squeeze the rest.

Sliding window made the same point with a different shape. Instead of squeezing from both ends, you expand a right boundary until the condition breaks, then shrink from the left until it holds again. The invariant is the same: you never need to revisit what you've already eliminated.

Binary search only clicked once I stopped thinking of it as "search in a sorted array" and started thinking of it as "cut the search space in half by asking a yes/no question at the midpoint." That reframe made rotated-array binary search (`search`) make sense without memorizing separate cases.

## Why it matters for what's ahead

DSA shows up in ML infrastructure more than the algorithms themselves. Efficient nearest-neighbor search (FAISS, HNSW) is applied binary search and heap operations. Backtracking underlies hyperparameter search. Graph algorithms (BFS, DFS, topological sort) appear in dependency resolution and DAG-based computation graphs. Knowing the patterns means recognizing them when they reappear in disguise.

## Files

- `*.py` — from-scratch implementations, one file per pattern
- `verify.py` — diffs implemented functions against brute-force / library references
- `notes/` — handwritten notes from source material, one PDF per pattern

## How to run

```bash
python <pattern>.py   # run a specific pattern
python verify.py      # verify all implemented functions
```

## Sources

- NeetCode.io — Blind 75 problems
- LeetCode problems

## Status

> Core patterns (two pointer, sliding window, binary search, top-k, slow-fast) done and verified. Tree, graph, backtracking stubs written. DP, monotonic stack, intervals, bit manipulation not started.

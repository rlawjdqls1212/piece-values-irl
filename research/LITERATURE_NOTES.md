# Literature used in the expanded introduction and related work

Primary sources checked on 2026-09-27. The narrative distinguishes behavior fitting, neural probing, removal-based evaluation labels, forward reinforcement learning, and reward identification. No claim of a first contextual chess value table or a new IRL algorithm is made.

- Ng & Russell (ICML 2000), *Algorithms for Inverse Reinforcement Learning*: https://dl.acm.org/doi/10.5555/645529.657801 . Publisher metadata confirmed; the scanned author PDF's extracted text was unusable, so no detailed theorem was attributed to that extraction.
- Abbeel & Ng (ICML 2004), *Apprenticeship Learning via Inverse Reinforcement Learning*: https://icml.cc/Conferences/2004/proceedings/papers/335.pdf . Feature-expectation matching and behavior guarantees are distinguished from exact reward recovery.
- Ziebart et al. (AAAI 2008): https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html . The original sequential maximum-entropy model is distinguished from the implemented horizon-one specialization.
- David et al. (GECCO 2009): https://arxiv.org/abs/1711.06840 . Move-based evaluation tuning predates this work; 2017 is the arXiv posting year, not the conference year.
- David, Netanyahu & Wolf (ICANN 2016), *DeepChess*: https://www.cs.tau.ac.il/~wolf/papers/deepchess.pdf . Conference/year/pages also checked against the publisher's ICANN 2016 volume: https://link.springer.com/book/10.1007/978-3-319-44781-0 . Position-comparison learning is not the same target as the present coefficient audit.
- Silver et al. (2017): https://arxiv.org/abs/1712.01815 . Forward self-play reinforcement learning and search are conceptually distinct from fitting an inverse reward to fixed demonstrations.
- McIlroy-Young et al. (KDD 2020), *Maia*: https://arxiv.org/abs/2006.01855 . Human move fidelity is distinct from engine playing strength.
- Ho & Ermon (NeurIPS 2016): https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html . Conceptual comparison only; no GAIL experiment is claimed.
- Kim et al. (ICML 2021): https://proceedings.mlr.press/v139/kim21c.html . Reward identification is separate from uniqueness of a ridge-penalized numerical optimum.
- Pálsson & Björnsson (IJCAI 2023): https://www.ijcai.org/proceedings/2023/541 . Neural concept probing and material values across game phases are prior work.
- Tang et al. (2026), *PAWN*: https://arxiv.org/abs/2604.15585 . Engine-derived piece-removal labels differ from demonstrated legal-move labels.
- Stockfish official UCI documentation: https://official-stockfish.github.io/docs/stockfish-wiki/UCI-%26-Commands.html . UCI_Elo settings are not measured human ratings under a 5,000-node query budget. The installed Stockfish 18 executable was also queried directly for its supported options.

The searches used the Exa Search skill. Only the above primary sources support the final manuscript; unrelated search results were discarded.

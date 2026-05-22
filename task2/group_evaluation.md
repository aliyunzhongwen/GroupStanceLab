# Community Detection and Group Quality Assessment Report

## 1. Overview

- Dataset: Twitter Political Interaction Network
- Nodes: 877
- Edges: 22099
- Party distribution: Democrat=471, Republican=406

## 2. Algorithm Results Comparison

| Method | Params | #Communities | Modularity | NMI | Purity | Max Size | Min Size |
|--------|--------|:------------:|:----------:|:---:|:------:|:--------:|:--------:|
| Louvain(res=0.5) | resolution=0.5 | 67 | 0.4909 | 0.2706 | 0.9339 | 164 | 1 |
| Louvain(res=0.8) | resolution=0.8 | 31 | 0.5636 | 0.3861 | 0.9316 | 267 | 1 |
| Louvain(res=1.0) | resolution=1.0 | 37 | 0.5673 | 0.3547 | 0.9327 | 228 | 1 |
| Louvain(res=1.2) | resolution=1.2 | 39 | 0.5621 | 0.3337 | 0.9339 | 187 | 1 |
| Louvain(res=1.5) | resolution=1.5 | 40 | 0.5488 | 0.3103 | 0.9361 | 154 | 1 |
| Label Propagation | default (best of 5 trials) | 27 | 0.0011 | 0.0452 | 0.5530 | 851 | 1 |
| Spectral(K=2) | K=2 | 2 | 0.0006 | 0.0000 | 0.5371 | 875 | 2 |
| Spectral(K=3) | K=3 | 3 | 0.4227 | 0.5894 | 0.9088 | 531 | 2 |
| Spectral(K=4) | K=4 | 4 | 0.4293 | 0.5901 | 0.9122 | 526 | 2 |
| Spectral(K=5) | K=5 | 5 | 0.4367 | 0.5756 | 0.9179 | 517 | 2 |
| Spectral(K=6) | K=6 | 6 | 0.5200 | 0.4537 | 0.9168 | 370 | 2 |
| Spectral(K=8) | K=8 | 8 | 0.5550 | 0.3885 | 0.9133 | 292 | 2 |
| Spectral(K=10) | K=10 | 10 | 0.5561 | 0.3968 | 0.9293 | 287 | 2 |
| KMeans(K=2) | K=2 | 2 | 0.3362 | 0.3882 | 0.8472 | 440 | 437 |
| KMeans(K=3) | K=3 | 3 | 0.3362 | 0.3544 | 0.8369 | 416 | 148 |
| KMeans(K=4) | K=4 | 4 | 0.2852 | 0.2962 | 0.8278 | 320 | 106 |
| KMeans(K=5) | K=5 | 5 | 0.2665 | 0.2647 | 0.8233 | 299 | 87 |
| KMeans(K=6) | K=6 | 6 | 0.1415 | 0.2479 | 0.8255 | 265 | 47 |
| KMeans(K=8) | K=8 | 8 | 0.1505 | 0.2459 | 0.8438 | 236 | 7 |
| KMeans(K=10) | K=10 | 10 | 0.1341 | 0.2243 | 0.8529 | 157 | 6 |
| Hybrid(a=0.3,K=2) | alpha=0.3, K=2 | 2 | 0.3309 | 0.3808 | 0.8461 | 454 | 423 |
| Hybrid(a=0.3,K=3) | alpha=0.3, K=3 | 3 | 0.3349 | 0.3345 | 0.8255 | 408 | 164 |
| Hybrid(a=0.3,K=4) | alpha=0.3, K=4 | 4 | 0.0618 | 0.2297 | 0.7742 | 318 | 147 |
| Hybrid(a=0.3,K=5) | alpha=0.3, K=5 | 5 | 0.0595 | 0.1960 | 0.7628 | 272 | 49 |
| Hybrid(a=0.3,K=6) | alpha=0.3, K=6 | 6 | 0.0493 | 0.1553 | 0.7275 | 234 | 46 |
| Hybrid(a=0.3,K=8) | alpha=0.3, K=8 | 8 | 0.0798 | 0.1088 | 0.7081 | 243 | 44 |
| Hybrid(a=0.5,K=2) | alpha=0.5, K=2 | 2 | 0.3309 | 0.3808 | 0.8461 | 454 | 423 |
| Hybrid(a=0.5,K=3) | alpha=0.5, K=3 | 3 | 0.3348 | 0.3335 | 0.8244 | 408 | 165 |
| Hybrid(a=0.5,K=4) | alpha=0.5, K=4 | 4 | 0.0820 | 0.2415 | 0.7754 | 328 | 141 |
| Hybrid(a=0.5,K=5) | alpha=0.5, K=5 | 5 | 0.0487 | 0.1949 | 0.7640 | 264 | 57 |
| Hybrid(a=0.5,K=6) | alpha=0.5, K=6 | 6 | 0.0458 | 0.1593 | 0.7298 | 235 | 46 |
| Hybrid(a=0.5,K=8) | alpha=0.5, K=8 | 8 | 0.0792 | 0.1079 | 0.7013 | 241 | 44 |
| Hybrid(a=0.7,K=2) | alpha=0.7, K=2 | 2 | 0.3355 | 0.3873 | 0.8483 | 450 | 427 |
| Hybrid(a=0.7,K=3) | alpha=0.7, K=3 | 3 | 0.3350 | 0.3344 | 0.8244 | 409 | 164 |
| Hybrid(a=0.7,K=4) | alpha=0.7, K=4 | 4 | 0.0929 | 0.2436 | 0.7754 | 328 | 137 |
| Hybrid(a=0.7,K=5) | alpha=0.7, K=5 | 5 | 0.0553 | 0.1962 | 0.7640 | 275 | 53 |
| Hybrid(a=0.7,K=6) | alpha=0.7, K=6 | 6 | 0.0465 | 0.1570 | 0.7286 | 227 | 46 |
| Hybrid(a=0.7,K=8) | alpha=0.7, K=8 | 8 | 0.0813 | 0.1107 | 0.7070 | 240 | 43 |

## 3. Best Method

**Method**: Spectral(K=4)

**Params**: K=4

**Metrics**:
- Modularity: 0.4293
- NMI: 0.5901
- Purity: 0.9122
- Communities: 4
- Overall score: 0.6597
- Size distribution: [526, 347, 2, 2]

## 4. Selection Criteria

Score = Modularity*0.35 + NMI*0.30 + Community_count_rationality*0.15 + Purity*0.20

Priority:
1. High modularity (reasonable community structure)
2. High NMI (correspondence with party labels)
3. Reasonable community count (4-10 preferred)
4. High purity (but not requiring 100%)

## 5. Conclusions

Key findings:

1. **Louvain** excels in modularity, auto-determines community count
2. **Spectral Clustering** offers flexible control over community count
3. **K-means** reflects node distribution in feature space
4. **Hybrid method** combines graph structure and node features
5. **Label Propagation** is fast but unstable

Final selection: **Spectral(K=4)**
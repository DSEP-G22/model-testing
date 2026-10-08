# Hand written generalization: XLNet and Laya vs every intent model

Sets, perturbations and scoring from `v3/docs/final/experiments/generalization.py`.

| Model | In-dist F1 | Hand written F1 | Hand written lenient acc | Drop | Padded F1 | Noise 10% F1 | Noise 20% F1 | Unknown AUROC | Unknown accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| laya_zeroshot | 0.907 | **0.715** | 0.806 | 0.192 | 0.886 | 0.667 | 0.454 | 0.900 | 40% |
| minilm_logreg | 0.992 | **0.680** | 0.806 | 0.312 | 0.963 | 0.762 | 0.515 | 0.996 | 0% |
| setfit_k64 | 0.969 | **0.657** | 0.769 | 0.311 | 0.953 | 0.679 | 0.446 | 0.995 | 0% |
| setfit_k8 | 0.927 | **0.609** | 0.731 | 0.318 | 0.901 | 0.609 | 0.395 | 0.991 | 0% |
| minilm_ft | 0.997 | **0.574** | 0.704 | 0.423 | 0.980 | 0.843 | 0.630 | 0.997 | 0% |
| xlnet_ft | 0.997 | **0.552** | 0.685 | 0.445 | 0.813 | 0.870 | 0.653 | 0.998 | 0% |
| minilm_zeroshot | 0.728 | **0.530** | 0.639 | 0.198 | 0.658 | 0.465 | 0.283 | 0.913 | 47% |
| laya_encoder_logreg | 0.974 | **0.489** | 0.611 | 0.486 | 0.285 | 0.582 | 0.306 | 0.943 | 30% |
| distilbert_ft | 0.996 | **0.418** | 0.546 | 0.578 | 0.952 | 0.831 | 0.611 | 0.997 | 0% |
| tfidf_logreg | 0.991 | **0.409** | 0.472 | 0.582 | 0.987 | 0.799 | 0.551 | 0.989 | 7% |
| tfidf_svc | 0.999 | **0.381** | 0.472 | 0.618 | 0.989 | 0.960 | 0.864 | 1.000 | 0% |

Hand written = 108 sentences, 4 per intent. Lenient counts 10 confusable intent pairs as right.
Unknown accepted = share of 30 out-of-taxonomy ISP messages whose confidence clears the 5th percentile
of in-distribution confidence (lower is better).

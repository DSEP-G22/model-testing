# Model benchmark summary

Environment: `{"python": "3.13.12", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.13.0+cu132", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "vram_gb": 6.0}`

## intent_classification

| model                       | family      |   accuracy |   macro_f1 |   train_s |   infer_ms |   size_mb |
|:----------------------------|:------------|-----------:|-----------:|----------:|-----------:|----------:|
| tfidf_word+char+linsvc      | traditional |     0.9992 |     0.9991 |    8.8874 |     1.4377 |      1.96 |
| minilm_finetuned            | deep        |     0.9969 |     0.9966 |   59.3847 |     8.2818 |     87.38 |
| distilbert_finetuned        | deep        |     0.9961 |     0.9956 |  126.002  |     8.1864 |    256.18 |
| minilm+linsvc               | embedding   |     0.9953 |     0.9946 |   11.9509 |     9.8514 |      0.08 |
| minilm+mlp256               | embedding   |     0.9939 |     0.9933 |    6.6988 |     9.8514 |      1.12 |
| minilm+logreg               | embedding   |     0.9931 |     0.9923 |   34.183  |     9.8514 |      0.04 |
| tfidf_word+logreg           | traditional |     0.9922 |     0.9912 |    1.1488 |     0.5548 |      0.92 |
| textcnn_scratch             | deep        |     0.9911 |     0.9902 |   22.6477 |     2.136  |      1.4  |
| tfidf_word+sgd_hinge        | traditional |     0.9897 |     0.9885 |    0.6914 |     0.8818 |      0.2  |
| minilm+knn5_cosine          | embedding   |     0.9892 |     0.9883 |    0.0386 |     9.8514 |     22.93 |
| tfidf_svd300+histgb         | traditional |     0.9797 |     0.9784 |   87.7557 |    38.7457 |     13.13 |
| tfidf_word+complementNB     | traditional |     0.9797 |     0.9781 |    0.2747 |     0.6197 |      1    |
| tfidf_word+randomforest     | traditional |     0.9786 |     0.9764 |    3.8598 |    83.9167 |     55.97 |
| minilm_prototype_all        | embedding   |     0.96   |     0.9553 |    0      |     9.7514 |    nan    |
| minilm_prototype_k25        | embedding   |     0.9553 |     0.9508 |    0      |     9.7514 |    nan    |
| minilm_prototype_k50        | embedding   |     0.9559 |     0.9506 |    0      |     9.7514 |    nan    |
| minilm_prototype_k10        | embedding   |     0.9195 |     0.9106 |    0      |     9.7514 |    nan    |
| minilm_prototype_k5         | embedding   |     0.8768 |     0.8662 |    0      |     9.7514 |    nan    |
| minilm_zeroshot_label_match | embedding   |     0.7386 |     0.7281 |    0      |     9.7514 |    nan    |
| minilm_prototype_k1         | embedding   |     0.7048 |     0.6836 |    0      |     9.7514 |    nan    |
| dummy_stratified            | traditional |     0.0369 |     0.0349 |  nan      |   nan      |    nan    |
| dummy_most_frequent         | traditional |     0.0416 |     0.003  |  nan      |   nan      |    nan    |

## intent_robustness_shift

| model                       | family      |   accuracy |   macro_f1 |   train_s | infer_ms   | size_mb   |
|:----------------------------|:------------|-----------:|-----------:|----------:|:-----------|:----------|
| tfidf_word+char+linsvc      | traditional |     0.9949 |     0.9943 |    2.7337 |            |           |
| minilm+logreg               | embedding   |     0.9794 |     0.9779 |    0.4791 |            |           |
| tfidf_word+logreg           | traditional |     0.9772 |     0.9749 |    0.7774 |            |           |
| distilbert_finetuned        | deep        |     0.9631 |     0.9582 |   45.3981 |            |           |
| minilm_zeroshot_label_match | embedding   |     0.7264 |     0.7137 |    0      |            |           |

## router_image_classification

| model                            | family      |   accuracy |   macro_f1 |   train_s |   infer_ms |   size_mb |
|:---------------------------------|:------------|-----------:|-----------:|----------:|-----------:|----------:|
| efficientnet_b0_frozen+logreg    | embedding   |     0.9704 |     0.9774 |    2.7617 |   nan      |      0.09 |
| efficientnet_b0_finetuned        | deep        |     0.963  |     0.9728 |  429.057  |    14.2546 |     15.65 |
| resnet18_frozen+logreg           | embedding   |     0.9481 |     0.9567 |    8.1098 |   nan      |      0.04 |
| mobilenet_v3_small_finetuned     | deep        |     0.963  |     0.9488 |  363.33   |     9.4293 |      5.98 |
| mobilenet_v3_small_frozen+logreg | embedding   |     0.9556 |     0.9456 |    3.5228 |   nan      |      0.07 |
| resnet18_finetuned               | deep        |     0.963  |     0.8999 |  377.228  |     5.3865 |     42.74 |
| colourhist+randomforest          | traditional |     0.8667 |     0.7723 |    1.33   |   102.039  |      9.11 |
| hog+colour+pca200+histgb         | traditional |     0.8519 |     0.7315 |   24.1052 |    38.7399 |      4.81 |
| pixels32+pca128+logreg           | traditional |     0.7556 |     0.6579 |   78.4534 |     0.4082 |      1.41 |
| hog+logreg                       | traditional |     0.7704 |     0.6409 |    4.4029 |     0.3068 |      0.3  |
| hog+linsvc                       | traditional |     0.7481 |     0.6366 |  175.156  |     0.3108 |      0.51 |
| dummy_most_frequent              | traditional |     0.2815 |     0.0366 |  nan      |   nan      |    nan    |

## voice_asr

| model                | family   |    wer |    cer |    rtf |   lang_id_acc | train_s   | infer_ms   | size_mb   |
|:---------------------|:---------|-------:|-------:|-------:|--------------:|:----------|:-----------|:----------|
| faster_whisper_small | asr      | 0.3897 | 0.2674 | 0.0876 |             1 |           |            |           |
| whisper_small        | asr      | 0.4299 | 0.2931 | 0.1299 |             1 |           |            |           |
| whisper_base         | asr      | 0.5974 | 0.3874 | 0.1027 |             1 |           |            |           |
| whisper_tiny         | asr      | 0.7605 | 0.5908 | 0.0911 |             1 |           |            |           |

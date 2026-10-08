# Model benchmark summary

Environment: `{"python": "3.13.12", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.13.0+cu132", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "vram_gb": 6.0}`

## intent_classification

| model                       | family      |   accuracy |   macro_f1 |   train_s |   infer_ms |   size_mb |
|:----------------------------|:------------|-----------:|-----------:|----------:|-----------:|----------:|
| tfidf_word+char+linsvc      | traditional |     0.9992 |     0.9991 |    8.0129 |     1.5223 |      1.96 |
| minilm_finetuned            | deep        |     0.9969 |     0.9966 |   46.8779 |     5.522  |     87.38 |
| distilbert_finetuned        | deep        |     0.9961 |     0.9956 |   82.8707 |     6.1289 |    256.18 |
| minilm+linsvc               | embedding   |     0.9953 |     0.9946 |    7.6898 |     6.1615 |      0.08 |
| minilm+mlp256               | embedding   |     0.9939 |     0.9933 |    3.3828 |     6.1615 |      1.12 |
| minilm+logreg               | embedding   |     0.9931 |     0.9923 |    0.794  |     6.1615 |      0.04 |
| tfidf_word+logreg           | traditional |     0.9922 |     0.9912 |    0.9195 |     0.627  |      0.92 |
| textcnn_scratch             | deep        |     0.9911 |     0.9902 |   12.068  |     1.1407 |      1.4  |
| tfidf_word+sgd_hinge        | traditional |     0.9897 |     0.9885 |    0.7595 |     1.0092 |      0.2  |
| minilm+knn5_cosine          | embedding   |     0.9892 |     0.9883 |    0.0354 |     6.1615 |     22.93 |
| tfidf_svd300+histgb         | traditional |     0.9797 |     0.9784 |   42.2924 |    27.5035 |     13.13 |
| tfidf_word+complementNB     | traditional |     0.9797 |     0.9781 |    0.3177 |     1.1054 |      1    |
| tfidf_word+randomforest     | traditional |     0.9786 |     0.9764 |    4.0672 |    69.5555 |     55.97 |
| minilm_prototype_all        | embedding   |     0.96   |     0.9553 |    0      |     6.0615 |    nan    |
| minilm_prototype_k25        | embedding   |     0.9553 |     0.9508 |    0      |     6.0615 |    nan    |
| minilm_prototype_k50        | embedding   |     0.9559 |     0.9506 |    0      |     6.0615 |    nan    |
| minilm_prototype_k10        | embedding   |     0.9195 |     0.9106 |    0      |     6.0615 |    nan    |
| minilm_prototype_k5         | embedding   |     0.8768 |     0.8662 |    0      |     6.0615 |    nan    |
| minilm_zeroshot_label_match | embedding   |     0.7386 |     0.7281 |    0      |     6.0615 |    nan    |
| minilm_prototype_k1         | embedding   |     0.7048 |     0.6836 |    0      |     6.0615 |    nan    |
| dummy_stratified            | traditional |     0.0369 |     0.0349 |  nan      |   nan      |    nan    |
| dummy_most_frequent         | traditional |     0.0416 |     0.003  |  nan      |   nan      |    nan    |

## intent_latest_models

| model                    | family    |   accuracy |   macro_f1 |   train_s |   infer_ms |   size_mb |
|:-------------------------|:----------|-----------:|-----------:|----------:|-----------:|----------:|
| xlnet_finetuned          | deep      |     0.9975 |     0.9974 |  339.112  |    16.4475 |     449.9 |
| laya_encoder+logreg      | embedding |     0.9767 |     0.9742 |    0.7629 |    25.5167 |     nan   |
| laya_zeroshot            | embedding |     0.9145 |     0.9073 |    0      |    76.9077 |     nan   |
| laya_zeroshot_shortlist8 | embedding |     0.9054 |     0.8957 |    0      |    55.0672 |     nan   |

## intent_robustness_shift

| model                       | family      |   accuracy |   macro_f1 |   train_s | infer_ms   | size_mb   |
|:----------------------------|:------------|-----------:|-----------:|----------:|:-----------|:----------|
| tfidf_word+char+linsvc      | traditional |     0.9949 |     0.9943 |    2.3289 |            |           |
| minilm+logreg               | embedding   |     0.9794 |     0.9779 |    1.0626 |            |           |
| tfidf_word+logreg           | traditional |     0.9772 |     0.9749 |    0.621  |            |           |
| distilbert_finetuned        | deep        |     0.9631 |     0.9582 |   44.5327 |            |           |
| minilm_zeroshot_label_match | embedding   |     0.7264 |     0.7137 |    0      |            |           |

## priority_latest_models

| model                | family    |   accuracy |   macro_f1 |   train_s |   infer_ms | size_mb   |
|:---------------------|:----------|-----------:|-----------:|----------:|-----------:|:----------|
| xlnet_finetuned      | deep      |     0.7467 |     0.7557 |   61.9257 |    19.1717 |           |
| laya_encoder+logreg  | embedding |     0.6933 |     0.7063 |    0.1737 |    28.1542 |           |
| distilbert_finetuned | deep      |     0.6889 |     0.6585 |   27.1898 |     6.0048 |           |
| laya_zeroshot        | embedding |     0.3667 |     0.2946 |    0      |    46.378  |           |

## router_image_classification

| model                            | family      |   accuracy |   macro_f1 |   train_s |   infer_ms |   size_mb |
|:---------------------------------|:------------|-----------:|-----------:|----------:|-----------:|----------:|
| efficientnet_b0_frozen+logreg    | embedding   |     0.9704 |     0.9774 |    0.1552 |   nan      |      0.09 |
| efficientnet_b0_finetuned        | deep        |     0.963  |     0.9728 |  405.509  |    12.6514 |     15.65 |
| resnet18_frozen+logreg           | embedding   |     0.9481 |     0.9567 |    0.2803 |   nan      |      0.04 |
| mobilenet_v3_small_finetuned     | deep        |     0.963  |     0.9488 |  344.928  |     8.5433 |      5.98 |
| mobilenet_v3_small_frozen+logreg | embedding   |     0.9556 |     0.9456 |    0.1343 |   nan      |      0.07 |
| resnet18_finetuned               | deep        |     0.963  |     0.8999 |  377.24   |     4.5667 |     42.74 |
| colourhist+randomforest          | traditional |     0.8667 |     0.7723 |    0.9912 |    80.0826 |      9.11 |
| hog+colour+pca200+histgb         | traditional |     0.8519 |     0.7315 |   12.6744 |    25.3107 |      4.81 |
| pixels32+pca128+logreg           | traditional |     0.7556 |     0.6579 |    1.1203 |     0.3044 |      1.41 |
| hog+logreg                       | traditional |     0.7704 |     0.6409 |    0.3396 |     0.2821 |      0.3  |
| hog+linsvc                       | traditional |     0.7481 |     0.6366 |  143.46   |     0.2902 |      0.51 |
| dummy_most_frequent              | traditional |     0.2815 |     0.0366 |  nan      |   nan      |    nan    |

## voice_asr

| model                      | family   |    wer |    cer |    rtf |   lang_id_acc | train_s   | infer_ms   | size_mb   |
|:---------------------------|:---------|-------:|-------:|-------:|--------------:|:----------|:-----------|:----------|
| faster_whisper_small       | asr      | 0.1838 | 0.139  | 0.0837 |             1 |           |            |           |
| whisper_small              | asr      | 0.1892 | 0.1437 | 0.099  |             1 |           |            |           |
| whisper_base               | asr      | 0.2925 | 0.21   | 0.0661 |             1 |           |            |           |
| whisper_tiny               | asr      | 0.3657 | 0.2548 | 0.0409 |             1 |           |            |           |
| whisper_small_translate_en | asr      | 0.7032 | 0.5442 | 0.1619 |           nan |           |            |           |

# Fine-Tuning Pre-trained Transformer Models for Sentiment Analysis

This repository contains a comparative sentiment analysis project based on **fine-tuning two pre-trained transformer models** on the **IMDb movie review dataset**:

- **BERT-base** (`google-bert/bert-base-uncased`)
- **DistilBERT** (`distilbert/distilbert-base-uncased`)

The goal of the project is to compare the two models under the **same experimental setup** and examine the trade-off between **classification performance** and **training efficiency**.

## Project Scope

The experimental part of the project focuses on **two models only**:

1. **BERT-base**
2. **DistilBERT**

The accompanying report also includes a **theoretical discussion of ModernBERT**, but **ModernBERT was not fine-tuned in the notebooks in this repository**. This README therefore documents only what is implemented and evaluated in the code.

## Dataset

The models were fine-tuned on the **IMDb dataset** from Hugging Face (`stanfordnlp/imdb`).

Dataset setup used in both notebooks:
- Original training set: **25,000** reviews
- Test set: **25,000** reviews
- The original training split was further divided into:
  - **22,500** training examples
  - **2,500** validation examples
- Split performed with `train_test_split(test_size=0.1, seed=42)`

This setup is identical in both experiments to ensure a fair comparison.

## Models

### 1. BERT-base
- Model: `google-bert/bert-base-uncased`
- Type: encoder-based transformer for sequence classification
- Fine-tuned for binary sentiment classification

### 2. DistilBERT
- Model: `distilbert/distilbert-base-uncased`
- Type: distilled / lighter version of BERT
- Fine-tuned for binary sentiment classification

## Methodology

Both models were trained with the **same pipeline and training configuration**.

### Preprocessing
- Tokenization performed with each model's corresponding tokenizer
- Maximum sequence length: **512 tokens**
- Truncation enabled
- Padding set to **`max_length`** in the tokenization step
- Original `text` column removed after tokenization
- Datasets converted to **PyTorch tensors** with `set_format("torch")`

### Training setup
- Framework: **Hugging Face Transformers**
- Training API: `Trainer`
- Number of labels: **2**
- Epochs: **2**
- Train batch size: **8**
- Evaluation batch size: **8**
- Evaluation strategy: **every epoch**
- Save strategy: **every epoch**
- Best model loaded at the end based on **accuracy**
- Mixed precision enabled with **`fp16=True`**
- `report_to="none"`

### Metrics
Two evaluation metrics were used on the **test set**:
- **Accuracy**
- **F1-score**

Accuracy was used during model selection, while F1-score was computed in a separate evaluation step after training.

## Results

### Test set performance

| Model | Accuracy | F1-score |
|---|---:|---:|
| BERT-base | **0.93608** | **0.93663** |
| DistilBERT | 0.93104 | 0.93085 |

### Training runtime recorded in the notebooks

| Model | Training runtime |
|---|---:|
| BERT-base | **1333.77 s** |
| DistilBERT | **665.72 s** |

## Interpretation

The results show that **both models achieve strong sentiment classification performance** on IMDb.

- **BERT-base** achieved the best overall performance with slightly higher Accuracy and F1-score.
- **DistilBERT** performed very closely to BERT-base, with only a small drop in performance.
- In the recorded notebook runs, **DistilBERT trained in about half the time**, making it a strong efficiency-oriented alternative.

This makes the comparison useful from a practical point of view:
- choose **BERT-base** when the highest possible performance is the priority
- choose **DistilBERT** when faster training and lower computational cost matter more

## Repository Contents

- `Βert_Base_finetuning.ipynb` — notebook for fine-tuning and evaluating BERT-base
- `Distill_Bert_finetuning.ipynb` — notebook for fine-tuning and evaluating DistilBERT
- `Report.pdf` — written report with methodology, results discussion, and theoretical comparison section

## Notes on the Notebooks

The notebooks were developed in **Google Colab** and include Google Drive mounting logic for saving outputs.

They also save evaluation metrics to JSON files under a user-defined `SAVE_DIR` during execution. These generated files are **not included in the zip**, but the final reported metrics are visible in the notebook outputs.

## Limitations

This repository reflects a **course / academic comparison project** and has some limitations:
- experiments are restricted to **2 epochs**
- no hyperparameter search is performed
- no error analysis is included
- no confusion matrix or class-wise breakdown is reported
- no additional datasets or cross-domain testing are used
- ModernBERT is discussed only theoretically, not experimentally

## Possible Future Improvements

- add **error analysis** on misclassified reviews
- compare against a **classical ML baseline** (e.g. Logistic Regression + TF-IDF)
- include **confusion matrices** and per-class precision/recall
- test a **smaller max sequence length** for faster training
- extend the comparison with **ModernBERT** or other encoder models
- convert the notebooks into a more reproducible script-based pipeline

## Conclusion

This project demonstrates a clean comparison between **BERT-base** and **DistilBERT** for sentiment analysis under the same training conditions. The experiments show that while **BERT-base** performs slightly better, **DistilBERT** offers nearly the same performance with substantially lower training time in the recorded runs.

For a sentiment classification task like IMDb, this makes **DistilBERT a very competitive practical choice**, especially when computational efficiency matters.

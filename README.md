
# 📚 MAP: Charting Student Math Misunderstandings

## Project Overview
This solution tackles the **Misconception Annotation Project (MAP)** challenge by predicting systematic student math errors from open-ended explanations. Our pipeline is engineered for maximum performance, robustness, and generalization, utilizing a **Single-Stage Multi-Label Classification** architecture combined with **K-Fold Cross-Validation** and **Model Ensembling**.

The primary goal is to achieve the highest possible **Mean Average Precision @ 3 (MAP@3)** score, the metric for the competition.

---

## 🎯 Final Methodology: Ensembled K-Fold Training

Our final approach is a **K-Fold Cross-Validation Ensemble**, which significantly mitigates overfitting and increases predictive stability compared to a single model.

### 1. Model & Base Architecture
* **Model:** **DeBERTa-v3 Base** (a powerful transformer optimized for mathematical reasoning and NLP tasks).
* **Approach:** Single-Stage Multi-Label Classification. The model directly outputs probabilities for all possible `Category:Misconception` labels.

### 2. Feature Engineering
A core optimization is the creation of a rich, concatenated input text to maximize context for the model:

$$
\text{Input} = \text{"Question: [QuestionText] \n Answer: [MC\_Answer] \n Explanation: [StudentExplanation]"}
$$

This structure ensures the model has the problem context, the student's final choice, and the reasoning all in one sequence.

### 3. K-Fold Cross-Validation (K=5)
* The training data is split into **5 non-overlapping folds** to ensure all data is used for both training and validation across the entire process.
* **Robustness:** By training 5 slightly different models, the final ensemble prediction is less susceptible to random noise and data bias than any single model.

### 4. Training Optimization (Colab + Drive)
* **Training Loop:** We used a custom loop (`run_kfold_pipeline`) to train one model per fold.
* **Storage:** Models are saved incrementally to Google Drive (`BASE_FOLD_DIR`), allowing for resume functionality and ensuring persistent storage, which is critical for long-running Kaggle/Colab training sessions.
* **Hardware Efficiency:** Training utilized **FP16** (mixed-precision) on the GPU to maximize the batch size and reduce training time.

---

## 🔑 Key Technical Details

### 1. Target Label & Encoding
* **Label Creation:** The final target is a concatenation of the two label columns: `Category:Misconception` (e.g., `Incorrect:Adding_across`).
* **Encoding:** `MultiLabelBinarizer` converts the list of true labels for each response into a numerical **one-hot vector** $\mathbf{Y} \in \{0, 1\}^{L}$, where $L$ is the total number of unique labels.

### 2. The MAP@3 Metric Implementation
The notebook includes a custom `map_at_k` function which is critical for two reasons:
* **Training Metric:** It is set as the `metric_for_best_model` in the `TrainingArguments`, directly optimizing the model for the competition's final metric.
* **Evaluation:** It ensures the competition's unique scoring rule (a correct label is only scored once per observation) is accurately computed.

### 3. Prediction & Ensembling
* **Model Ensemble:** After training all 5 models, the inference pipeline loads each checkpoint and generates predictions.
* **Averaging:** The **probabilities** from all 5 models are averaged (Model Averaging).
$$
\text{AvgProb} = \frac{1}{5} \sum_{i=1}^{5} \text{Sigmoid}(\text{Logits}_i)
$$
* **Final Prediction:** The **top-3 indices** from this averaged probability vector are selected, converted back to the `Category:Misconception` string, and saved to the submission file.

---

## 📊 Evaluation Metric: MAP@3

The Mean Average Precision at 3 (MAP@3) is the official evaluation metric. It measures the quality of the ranked predictions in the top 3 slots.

| Metric | Rationale |
| :--- | :--- |
| **Precision** | Correctly predicted labels must be at the highest ranks. |
| **@3 (Cutoff)** | Only the top 3 predictions are considered for scoring. |
| **Average** | The score is averaged across all predictions up to the rank where the relevant label is found. |

This metric rewards models that are highly confident and accurate in their top suggestions, which is essential for diagnostic tools where teachers need reliable, prioritized feedback.

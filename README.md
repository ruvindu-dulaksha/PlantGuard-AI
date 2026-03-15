# 🌿 PlantGuard AI — Sustainable Agriculture Advisor

> **Computer Vision Coursework — Individual Report**
> **Module:** NB6007CEM — Computer Vision
> **Student:** K.D. Ruvindu Dulaksha
> **Student ID:** COBSCCOMP4Y241P-018
> **Degree:** BSc (Hons) Computing — Batch BSCCOMP24.1P
> **Institution:** National Institute of Business Management (NIBM) — School of Computing and Engineering
> **Lecturer:** Dr. Kaneeka

---

## Demo Video

**Watch the full system demonstration on YouTube:** https://youtu.be/5NuNOEsbK6g

The demo video walks through the complete notebook (all 18 cells) and the live Streamlit web application, including real-time plant disease classification and Flan-T5 advisory generation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Installation and Requirements](#4-installation-and-requirements)
5. [How to Run the Notebook](#5-how-to-run-the-notebook)
6. [How to Run the Streamlit App](#6-how-to-run-the-streamlit-app)
7. [Technical Architecture](#7-technical-architecture)
8. [Model 1 — EfficientNet-B0](#8-model-1--efficientnet-b0)
9. [Model 2 — Vision Transformer ViT-B/16](#9-model-2--vision-transformer-vit-b16)
10. [Preprocessing and Data Augmentation](#10-preprocessing-and-data-augmentation)
11. [Loss Function — Cross-Entropy](#11-loss-function--cross-entropy)
12. [Backpropagation and Optimisation](#12-backpropagation-and-optimisation)
13. [Learning Rate Scheduling and Early Stopping](#13-learning-rate-scheduling-and-early-stopping)
14. [LLM Advisory System — Flan-T5](#14-llm-advisory-system--flan-t5)
15. [AGROVOC Ontology Integration](#15-agrovoc-ontology-integration)
16. [Evaluation Metrics and Results](#16-evaluation-metrics-and-results)
17. [Notebook Cell-by-Cell Guide](#17-notebook-cell-by-cell-guide)
18. [Known Issues and Fixes](#18-known-issues-and-fixes)
19. [References](#19-references)

---

## 1. Project Overview

PlantGuard AI is an end-to-end deep learning system for automated plant disease detection and agricultural advisory generation. Plant diseases cause 20-40% of global food production losses annually (FAO, 2021), with the most severe impact on smallholder farmers in developing countries who lack access to expert agronomists.

This system addresses that gap by combining:

- **Two transfer learning models** — EfficientNet-B0 (CNN) and Vision Transformer ViT-B/16 — trained on the PlantVillage dataset to classify 15 plant disease and healthy classes from leaf photographs.
- **A large language model (Flan-T5)** to convert classification results into structured, farmer-readable advisory text covering cause, remedy, prevention, and severity.
- **FAO AGROVOC ontology** cross-referencing to link every predicted disease to an internationally recognised agricultural concept identifier.
- **A Streamlit web application** providing a browser-based interface accessible without any software installation.

### Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | Epochs |
|---|---|---|---|---|---|
| EfficientNet-B0 | 99.26% | 99.26% | 99.26% | 99.26% | 10 (full) |
| Vision Transformer ViT-B/16 | 98.50% | 98.50% | 98.50% | 98.50% | 8 (early stop) |

Both models exceed the benchmark accuracy reported by Mohanty et al. (2016) using AlexNet on a comparable PlantVillage configuration (99.35%).

---

## 2. Project Structure

```
PlantGuard_Submission/
|
|-- README.md                                                  <- this file
|
|-- Sustainable_Agriculture_Plant_Health_Monitor_              <- main notebook
|   ruvindu_dulaksha.ipynb
|
|-- app.py                                                     <- Streamlit web app
|
|-- final_saved_models/                                        <- trained model weights
    |-- efficientnet_best.pth                                  <- EfficientNet-B0 weights
    |-- vit_best.pth                                           <- ViT-B/16 weights
    |-- class_names.json                                       <- 15 class label list
```

Note: The `final_saved_models/` folder is required to run the Streamlit app. Download it from the Colab notebook (Cell 17) after training, or use pre-trained weights if provided separately.

---

## 3. Dataset

**Dataset:** PlantVillage (Kaggle — emmarex/plantdisease)  
**Total Images:** approximately 20,636  
**Classes:** 15 (10 tomato diseases/healthy, 3 potato, 2 bell pepper)  
**Source:** Publicly available, open-access, no personal data involved  

### 15 Classes

| # | Class | Type | AGROVOC ID |
|---|---|---|---|
| 1 | Pepper Bell — Bacterial Spot | Disease | c_72352 |
| 2 | Pepper Bell — Healthy | Healthy | c_5873 |
| 3 | Potato — Early Blight | Disease | c_25158 |
| 4 | Potato — Late Blight | Disease | c_25159 |
| 5 | Potato — Healthy | Healthy | c_5873 |
| 6 | Tomato — Bacterial Spot | Disease | c_72352 |
| 7 | Tomato — Early Blight | Disease | c_25158 |
| 8 | Tomato — Late Blight | Disease | c_25159 |
| 9 | Tomato — Leaf Mold | Disease | c_4871 |
| 10 | Tomato — Septoria Leaf Spot | Disease | c_6925 |
| 11 | Tomato — Spider Mites | Disease | c_7345 |
| 12 | Tomato — Target Spot | Disease | c_4871 |
| 13 | Tomato — Yellow Leaf Curl Virus | Disease | c_92411 |
| 14 | Tomato — Mosaic Virus | Disease | c_92412 |
| 15 | Tomato — Healthy | Healthy | c_5873 |

### Data Split Strategy

| Split | Proportion | Samples | Purpose |
|---|---|---|---|
| Training | 70% | ~14,445 | Optimising model weights via backpropagation |
| Validation | 15% | ~3,095 | Hyperparameter tuning, early stopping, LR scheduling |
| Test | 15% | 3,096 | Final independent evaluation (never seen during training) |

All splits use stratified sampling to preserve the class distribution proportions across all three subsets. The test set is strictly isolated and was never accessed during training or model selection.

---

## 4. Installation and Requirements

### Python Version

Python 3.8 or higher recommended.

### Install all dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm scikit-learn seaborn matplotlib transformers pillow streamlit
```

### For Google Colab (GPU — recommended)

Run Cell 1 of the notebook. Colab provides a free T4 GPU (CUDA 11.8) which significantly reduces training time. Training on CPU is not recommended — EfficientNet takes approximately 2-3 hours and ViT takes approximately 5-6 hours on CPU.

### Library versions used

| Library | Purpose |
|---|---|
| torch 2.x (CUDA 11.8) | Deep learning framework and autograd engine |
| torchvision 0.15+ | Dataset loading, transforms, EfficientNet pretrained weights |
| timm 0.9+ | Vision Transformer pretrained weights via timm model hub |
| transformers 4.36+ | Flan-T5 LLM (T5ForConditionalGeneration, T5Tokenizer) |
| scikit-learn 1.3+ | Evaluation metrics — accuracy, precision, recall, F1 |
| seaborn 0.12+ | Confusion matrix heatmap visualisation |
| matplotlib 3.7+ | Training curves, class distribution chart, bar charts |
| Pillow 10.0+ | Image loading and conversion |
| streamlit 1.28+ | Browser-based web application interface |
| pandas 2.0+ | Model comparison table and CSV export |

---

## 5. How to Run the Notebook

### Step-by-step guide for Google Colab

1. Open https://colab.research.google.com
2. Go to File > Upload notebook > upload the .ipynb file
3. Go to Runtime > Change runtime type > GPU (T4)
4. Run Cell 1 — installs all libraries
5. Run Cell 2 — mounts Google Drive (optional but recommended for saving)
6. Run Cell 3 — imports all Python modules
7. Run Cell 4 — confirms GPU availability (should print "Using device: cuda")
8. Run Cell 5 — downloads and unzips the PlantVillage dataset via Kaggle API
9. Run Cell 6 — sets the dataset directory path variable
10. Run Cell 7 — loads dataset, prints class information, plots class distribution chart and sample images grid
11. Run Cell 8 — creates preprocessing pipelines and performs stratified 70/15/15 splits with DataLoaders
12. Run Cell 9 — defines the shared train_model() function with loss, optimiser, scheduler, and early stopping
13. Run Cell 10 — loads pretrained EfficientNet-B0, replaces final classifier, runs training for 10 epochs
14. Run Cell 11 — loads pretrained ViT-B/16 via timm, runs training (stops at epoch 8 via early stopping)
15. Run Cell 12 — evaluates both models on the test set: prints classification report and confusion matrices
16. Run Cell 13 — generates the side-by-side performance comparison bar chart
17. Run Cell 14 — loads Flan-T5 directly using T5ForConditionalGeneration and defines AGROVOC ontology map
18. Run Cell 15 — runs the full end-to-end pipeline on a single test image
19. Run Cell 16 — saves both model weight files and class_names.json to final_saved_models/
20. Run Cell 17 — zips and downloads the models folder to your local machine
21. Cell 18 — shows the streamlit run app.py command

Important: Always run cells in order from top to bottom. Do not skip Cell 9 (training function definition) before running Cells 10 or 11.

---

## 6. How to Run the Streamlit App

### Requirements before running

- app.py in your working directory
- final_saved_models/ folder in the same directory, containing:
  - efficientnet_best.pth
  - vit_best.pth
  - class_names.json

### Commands

```bash
# Install dependencies
pip install streamlit timm transformers torch torchvision pillow

# Launch the app
streamlit run app.py
```

The app opens automatically in your browser at http://localhost:8501

### App Features — 3 Tabs

**Tab 1 — Diagnose Plant**
- Upload any JPG or PNG leaf photograph
- Select model: EfficientNet-B0, Vision Transformer, or Both
- View predicted disease class, confidence percentage, AGROVOC ontology tag
- View disease details panel: cause, remedy, prevention, severity rating
- Read Flan-T5 generated advisory paragraph in plain language
- When "Both" selected: shows agreement or disagreement indicator between the two models

**Tab 2 — Model Comparison**
- Interactive performance dashboard with sliders pre-loaded with real test results
- Live-updating grouped bar chart comparing all four metrics side by side

**Tab 3 — Disease Database**
- Searchable, severity-filterable knowledge base for all 15 classes
- Each entry shows AGROVOC FAO ontology reference, cause, remedy, prevention
- Expandable cards for detailed viewing

---

## 7. Technical Architecture

```
Input Leaf Image (JPG/PNG)
         |
         v
  Preprocessing Pipeline
  Resize 224x224 + Normalize (ImageNet mean/std)
         |
    _____|_____
   |           |
   v           v
EfficientNet-B0    Vision Transformer ViT-B/16
~5.3M params       ~86M params
MBConv-SE blocks   14x14 patch grid
Local textures     Global attention
   |           |
   |___________|
         |
         v
  Softmax -> 15-class probability vector
  Disease class + confidence percentage
         |
         v
  AGROVOC Ontology Cross-Reference
  Class -> FAO concept identifier
         |
         v
  Flan-T5 LLM Advisory Generator
  Cause / Remedy / Prevention / Severity
         |
         v
  Streamlit Web Application
  3-tab browser interface
```

---

## 8. Model 1 — EfficientNet-B0

### What it is

EfficientNet-B0 is a Convolutional Neural Network developed by Google Brain (Tan and Le, 2019) using Neural Architecture Search (NAS). It uses compound scaling — simultaneously scaling network width, depth, and input resolution by a fixed ratio — making it highly parameter-efficient while maintaining strong accuracy.

### Architecture

- **Building block:** Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation (MBConv-SE)
- **Parameters:** approximately 5.3 million
- **Input size:** 224 x 224 x 3 RGB
- **Pretrained on:** ImageNet-1k (1.28 million images, 1,000 classes)

### How MBConv-SE works

Each MBConv block performs three operations in sequence:

1. **Expansion convolution (1x1)** — increases channel depth to create a wider feature space for richer representations
2. **Depthwise convolution (3x3 or 5x5)** — applies a separate filter to each channel independently, capturing spatial patterns while reducing computation
3. **Projection convolution (1x1)** — compresses channels back to original depth with a residual skip connection

The Squeeze-and-Excitation (SE) mechanism inside each block:
- **Squeeze:** Global average pooling collapses spatial dimensions into one value per channel
- **Excitation:** Two fully connected layers learn a weighting vector — which channels (colour or texture features) are most important for the current input
- **Scale:** Channel outputs are multiplied by the learned weights, amplifying diagnostic features and suppressing noise

This is why EfficientNet excels at plant disease classification — diseases appear as localised texture and colour anomalies (lesion rings, spots, mould patterns) that the SE channel attention mechanism learns to detect and amplify.

### Transfer Learning

```python
model_eff = models.efficientnet_b0(pretrained=True)
# Replace the final classifier: 1000 ImageNet classes -> 15 plant disease classes
model_eff.classifier[1] = nn.Linear(model_eff.classifier[1].in_features, num_classes)
```

The pretrained ImageNet weights provide low-level feature detectors (edges, textures, colour gradients) that transfer well to leaf disease features. Only the final classification layer is randomly initialised and learned from scratch. All other layers are fine-tuned with a conservative learning rate.

### Training Result

- Epochs: 10 (complete — no early stopping triggered)
- Final test accuracy: 99.26% across all four weighted metrics
- Validation loss consistently below training loss throughout — expected because augmentation is applied only to training data, making training artificially harder

---

## 9. Model 2 — Vision Transformer ViT-B/16

### What it is

Vision Transformer (ViT-B/16) was introduced by Dosovitskiy et al. (2020) at Google Brain. It applies the Transformer architecture — originally designed for natural language processing (Vaswani et al., 2017) — directly to images by treating fixed-size image patches as sequence tokens, analogous to words in a sentence.

### Architecture

- **Input processing:** 224x224 image divided into a 14x14 grid of 16x16 pixel patches, producing 196 patch tokens
- **Patch embedding:** Each patch is flattened and linearly projected into a 768-dimensional embedding vector
- **Class token:** A learnable [CLS] token is prepended to the sequence — its final state after all attention layers is used for classification
- **Positional encoding:** Learned positional embeddings are added to each token to preserve spatial order information
- **Transformer encoder:** 12 layers of Multi-Head Self-Attention (MHSA) followed by Feed-Forward Network (FFN) with LayerNorm
- **Parameters:** approximately 86 million
- **Pretrained on:** ImageNet-21k (14 million images, 21,841 classes)

### How Multi-Head Self-Attention works

For each patch token, the model computes three vectors — Query (Q), Key (K), and Value (V) — using learned weight matrices. Attention is then computed as:

```
Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V
```

Where:
- Q (Query) represents what the current token is looking for in other tokens
- K (Key) represents what information each token contains
- V (Value) represents the actual information to be aggregated
- sqrt(d_k) is a scaling factor to prevent vanishing gradients from large dot product values

With 12 attention heads, each head independently attends to different spatial relationships simultaneously — one head might focus on lesion boundaries, another on colour distribution across the leaf, another on margin curl patterns.

After each self-attention layer, a position-wise Feed-Forward Network applies two linear transformations with a GELU activation:

```
FFN(x) = GELU(x * W1 + b1) * W2 + b2
```

This global attention mechanism is particularly strong for plant diseases that affect the entire leaf surface, such as Tomato Yellow Leaf Curl Virus (whole-leaf curling and yellowing) and Tomato Mosaic Virus (diffuse mosaic discolouration), where EfficientNet's local receptive fields may miss global patterns.

### Transfer Learning

```python
model_vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
```

The timm library loads ImageNet-21k pretrained weights and automatically replaces the classification head with a new linear layer outputting num_classes = 15.

### Training Result

- Epochs: 8 (early stopping triggered — validation loss spike at epoch 5)
- Final test accuracy: 98.50% across all four weighted metrics
- ViT converges more slowly than CNN without CNN-specific inductive biases such as translational invariance and local connectivity. A lower learning rate (2e-5) with linear warm-up as recommended by the original paper would improve convergence. Despite the shared 1e-4 rate, 98.5% accuracy demonstrates strong performance.

---

## 10. Preprocessing and Data Augmentation

### Training Transform Pipeline

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Standard input size for pretrained models
    transforms.RandomHorizontalFlip(),      # Simulates different leaf orientations
    transforms.RandomVerticalFlip(),        # Further orientation variation
    transforms.RandomRotation(30),          # +-30 degrees rotation for field photography angles
    transforms.ColorJitter(                 # Simulates variable lighting and camera conditions
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.RandomAffine(               # Simulates camera angle and distance variation
        0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),                 # Converts PIL Image to float tensor in range [0,1]
    transforms.Normalize(                  # ImageNet normalisation — required for pretrained weights
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Why these augmentations?

PlantVillage images are taken under controlled laboratory conditions with plain white backgrounds and uniform overhead lighting. Real field photographs have variable angles, natural lighting, soil and sky backgrounds, and varying zoom levels. Augmentation bridges this domain gap by artificially increasing training set diversity and forcing models to learn disease features that are invariant to orientation and lighting — significantly improving generalisation to real-world images.

Augmentation is applied only to training data. Validation and test sets receive only resize and normalisation. This is why validation loss appears lower than training loss throughout EfficientNet training — training is artificially made harder by augmentation, not because of overfitting.

### Validation and Test Transform Pipeline

```python
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### ImageNet Normalisation Values

The mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] values are computed from the full ImageNet-1k dataset across the Red, Green, and Blue channels respectively. Both EfficientNet-B0 and ViT-B/16 were pretrained with inputs normalised to this distribution. Using different normalisation values would render the pretrained feature representations incompatible with the new input distribution.

---

## 11. Loss Function — Cross-Entropy

### Formula

For a single sample with true class y and model output logits z, the cross-entropy loss is:

```
Loss = -log( exp(z_y) / sum( exp(z_k) for all classes k ) )
     = -log( softmax(z)_y )
     = -z_y + log( sum( exp(z_k) ) )
```

For a batch of N samples:

```
L = -(1/N) * sum( log(p_{y_i}) )   for i = 1 to N
```

Where p_{y_i} is the softmax probability assigned to the correct class for sample i.

### How Softmax works

Before the loss is computed, raw model outputs (logits) are converted to a probability distribution:

```
softmax(z_j) = exp(z_j) / sum( exp(z_k) for all k )
```

This ensures all class probabilities are positive and sum to exactly 1.0, making them interpretable as confidence scores.

### Why Cross-Entropy for this task?

**Penalises confident wrong predictions heavily.** If the model predicts 99% probability for the wrong disease class, the loss is very large: -log(0.01) = 4.6. If it predicts 60% for the correct class, the loss is moderate: -log(0.6) = 0.51. This directly incentivises the model to be both correct and calibrated.

**Standard choice for mutually exclusive multi-class classification.** Each leaf image belongs to exactly one of 15 disease or healthy classes. Cross-entropy is the mathematically correct objective under this assumption.

**Numerically stable.** PyTorch's nn.CrossEntropyLoss internally combines LogSoftmax and Negative Log-Likelihood Loss using the log-sum-exp trick, preventing numerical overflow or underflow from very large or very small exponential values.

```python
criterion = nn.CrossEntropyLoss()
```

Model output logits are passed directly to criterion — no separate softmax layer is needed in the model definition.

---

## 12. Backpropagation and Optimisation

### What is Backpropagation?

Backpropagation is the algorithm that computes the gradient of the loss function with respect to every trainable parameter in the network. It works by applying the chain rule of calculus in reverse — propagating the error signal backwards from the output layer through each intermediate layer to the input.

For a network with parameters W at layer l and loss L:

```
dL/dW_l = dL/d(output_l) * d(output_l)/dW_l

Using the chain rule:
dL/dW_1 = dL/dL_n * dL_n/dL_{n-1} * ... * dL_2/dL_1 * dL_1/dW_1
```

Each gradient value tells us: if we increase this weight by a tiny amount, how much does the total loss increase (or decrease)? Parameters with large positive gradients need to decrease. Parameters with large negative gradients need to increase.

### Training Loop — One Epoch Step by Step

```python
# Step 1: Forward pass — compute predictions
outputs = model(images)          # shape: [batch_size, 15]

# Step 2: Compute loss
loss = criterion(outputs, labels)

# Step 3: Zero gradients — clear accumulated gradients from previous batch
optimizer.zero_grad()

# Step 4: Backward pass — compute gradients via backpropagation (chain rule)
loss.backward()

# Step 5: Update weights — gradient descent step using computed gradients
optimizer.step()
```

Without zero_grad() before each backward pass, gradients would accumulate across batches, causing incorrect weight updates. This is a common source of bugs.

### Optimiser — Adam (Adaptive Moment Estimation)

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

Adam improves over standard Stochastic Gradient Descent by maintaining two exponentially decaying moving averages per parameter:

**First moment (m_t)** — running average of gradients — estimates the direction of the gradient:
```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t        [beta1 = 0.9 by default]
```

**Second moment (v_t)** — running average of squared gradients — estimates the variance of the gradient:
```
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2       [beta2 = 0.999 by default]
```

Bias-corrected estimates (important in early training when m and v are close to zero):
```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

Final weight update rule:
```
theta_t = theta_{t-1} - lr * m_hat / ( sqrt(v_hat) + epsilon )   [epsilon = 1e-8]
```

The division by sqrt(v_hat) gives each parameter its own adaptive learning rate:
- Parameters with consistently large gradients get smaller effective learning rates (stable directions)
- Parameters with small or noisy gradients get larger effective learning rates (slow-moving directions need more encouragement)

### Why Adam for transfer learning?

This adaptive per-parameter learning rate is critical when fine-tuning pretrained weights. Early layers of EfficientNet and ViT contain carefully learned low-level feature detectors (edge detectors, colour filters, texture patterns from ImageNet). These layers need very small updates to preserve their useful representations. The new classification head needs larger updates to learn from scratch. Adam automatically adjusts these rates based on gradient history, without requiring manual per-layer learning rate configuration.

### Learning Rate: 1e-4

This conservative rate was chosen specifically for fine-tuning transfer learning:
- Too high (e.g. 1e-2): would destroy pretrained ImageNet feature representations in early epochs — catastrophic forgetting
- Too low (e.g. 1e-6): extremely slow convergence, impractical training time
- 1e-4: standard recommended rate for fine-tuning EfficientNet variants (Tan and Le, 2019). ViT-B/16 ideally uses 2e-5 with linear warm-up (Dosovitskiy et al., 2020) — the shared rate is a controlled experimental choice to ensure fair comparison between models.

---

## 13. Learning Rate Scheduling and Early Stopping

### ReduceLROnPlateau Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
```

The scheduler monitors validation loss at the end of every epoch. If validation loss does not decrease for patience=2 consecutive epochs, it multiplies the current learning rate by a reduction factor (default: 0.1).

This mechanism is particularly important for ViT training. At epoch 5, validation loss spiked from approximately 0.07 to 0.18 — indicating the learning rate was too high for the current stage of training. The scheduler reduced the learning rate, enabling finer weight updates in subsequent epochs and allowing the model to recover to a validation loss of approximately 0.065 by epoch 7.

### Early Stopping

```python
patience_limit = 3
counter = 0
best_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

# After each validation phase:
if val_loss < best_loss:
    best_loss = val_loss
    best_model_weights = copy.deepcopy(model.state_dict())  # Save best weights
    counter = 0
else:
    counter += 1
    if counter >= patience_limit:
        print("Early stopping triggered — restoring best weights")
        break

model.load_state_dict(best_model_weights)  # Always restore the best checkpoint
```

If validation loss does not improve for 3 consecutive epochs, training halts and the best-performing weights (from the epoch with lowest validation loss) are restored. This prevents the model from continuing to train after it has already reached its performance ceiling, avoiding overfitting to the training data.

- **EfficientNet-B0:** Early stopping never triggered. Trained for the full 10 epochs with continuously improving or stable validation loss.
- **ViT-B/16:** Early stopping triggered at epoch 8.

### Overfitting Detection

```python
if (train_acc - val_acc) > 0.15:
    print(f"WARNING: Possible overfitting — gap = {train_acc - val_acc:.2%}")
```

If training accuracy exceeds validation accuracy by more than 15 percentage points, a real-time warning is printed. Neither model triggered this warning, confirming no overfitting occurred in either training run.

---

## 14. LLM Advisory System — Flan-T5

### Model

Google Flan-T5-base is an instruction-tuned encoder-decoder Transformer model fine-tuned on over 1,800 NLP tasks (Chung et al., 2022). Its instruction-following capability makes it well-suited to structured prompt-based advisory text generation.

### Loading — Direct Load (Critical Fix)

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer  = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
llm_model  = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
llm_model.eval()
```

In Hugging Face Transformers version 4.36 and later, the pipeline("text2text-generation") abstraction was removed. The model must be loaded directly using T5ForConditionalGeneration and T5Tokenizer. This was a key technical challenge encountered and resolved during development. Using the old pipeline approach on current Transformers versions causes a KeyError at runtime.

### Inference Configuration

```python
outputs = llm_model.generate(
    **inputs,
    max_new_tokens=150,        # Maximum advisory paragraph length
    num_beams=4,               # Beam search — explores 4 candidate sequences simultaneously
    early_stopping=True,       # Halts beam search when all beams reach end-of-sequence token
    no_repeat_ngram_size=3     # Prevents repetitive 3-gram phrases in output
)
```

Beam search with num_beams=4 selects the output sequence with the highest overall joint probability across all candidate beams. This produces more coherent and complete text than greedy decoding, which only selects the most likely token at each individual step.

### Advisory Prompt Structure

```
Disease detected: {disease_name}

Please provide agricultural advice covering:
1. What causes this disease
2. How to treat it  
3. How to prevent future outbreaks
4. Severity level

Give practical advice for a farmer.
```

### Example Output

```
Disease: Tomato Late Blight
Advisory: Tomato Late Blight is caused by the oomycete pathogen Phytophthora infestans,
the same organism responsible for the Irish Potato Famine of 1845. Remove and destroy all
infected plant material immediately to prevent airborne spore spread. Apply copper-based
fungicides or chlorothalonil every 7-10 days during humid conditions. Ensure good air
circulation between plants and avoid overhead irrigation. This is a HIGH severity disease —
uncontrolled spread can destroy an entire crop field within days under warm, wet conditions.
```

---

## 15. AGROVOC Ontology Integration

AGROVOC is the Food and Agriculture Organisation's (FAO) international standard agricultural thesaurus — a multilingual, structured controlled vocabulary used for indexing and retrieving agricultural scientific literature in over 40 languages globally.

Every classification result produced by the system is cross-referenced with an official AGROVOC concept URI:

```
http://aims.fao.org/aos/agrovoc/c_XXXXX
```

### Complete Ontology Mapping (all 15 classes)

```python
AGROVOC_MAP = {
    "Pepper__bell___Bacterial_spot":            "c_72352",  # Bacterial leaf spot
    "Pepper__bell___healthy":                   "c_5873",   # Plant health
    "Potato___Early_blight":                    "c_25158",  # Alternaria solani
    "Potato___Late_blight":                     "c_25159",  # Phytophthora infestans
    "Potato___healthy":                         "c_5873",   # Plant health
    "Tomato_Bacterial_spot":                    "c_72352",  # Bacterial leaf spot
    "Tomato_Early_blight":                      "c_25158",  # Alternaria solani
    "Tomato_Late_blight":                       "c_25159",  # Phytophthora infestans
    "Tomato_Leaf_Mold":                         "c_4871",   # Leaf mould (Passalora fulva)
    "Tomato_Septoria_leaf_spot":                "c_6925",   # Septoria leaf spot
    "Tomato_Spider_mites_Two_spotted_spider_mite": "c_7345",# Spider mites
    "Tomato__Target_Spot":                      "c_4871",   # Leaf mould / target spot
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus":    "c_92411",  # TYLCV
    "Tomato__Tomato_mosaic_virus":              "c_92412",  # Tomato mosaic virus
    "Tomato___healthy":                         "c_5873",   # Plant health
}
```

### Why AGROVOC matters

Linking predictions to internationally standardised concept identifiers enables:
- Interoperability with global crop surveillance and early warning systems (e.g. EMPRES-i, GEOGLAM)
- Citation and referencing in peer-reviewed agricultural scientific literature
- Multi-language access — AGROVOC supports 40+ languages enabling use in non-English-speaking farming communities
- Traceability for regulatory, insurance, and research audit purposes
- Integration with extension officer knowledge management systems

---

## 16. Evaluation Metrics and Results

All metrics are computed exclusively on the held-out test set (3,096 samples — 15% of the full dataset) which was never accessed during training, validation, or model selection. Weighted averaging is applied to all metrics to account for class imbalance across the 15 classes.

### Metric Definitions

**Accuracy**
```
Accuracy = (TP + TN) / Total
```
Proportion of all test samples correctly classified. Straightforward but can be misleading for imbalanced datasets without weighted context.

**Precision (Weighted)**
```
Precision = TP / (TP + FP)
```
Of all samples the model predicted as a particular disease, what fraction actually had that disease. Low precision means unnecessary pesticide application costs and reduced farmer trust.

**Recall (Weighted)**
```
Recall = TP / (TP + FN)
```
Of all samples that actually had a particular disease, what fraction the model correctly detected. Low recall means missed infections, allowing disease to spread — directly causing crop loss. This is arguably the most important metric in an agricultural context.

**F1-Score (Weighted)**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall. The harmonic mean penalises extreme imbalance between the two, making F1 the most balanced single metric for imbalanced multi-class tasks.

### Full Results

| Metric | EfficientNet-B0 | Vision Transformer |
|---|---|---|
| Accuracy | 0.9926 (99.26%) | 0.9850 (98.50%) |
| Precision (weighted) | 0.9926 | 0.9850 |
| Recall (weighted) | 0.9926 | 0.9850 |
| F1-Score (weighted) | 0.9926 | 0.9850 |
| Training epochs | 10 (full) | 8 (early stop) |
| Parameters | ~5.3M | ~86M |
| Overfitting | None | None |

### Notable Per-Class Results — EfficientNet-B0

Classes with perfect F1 = 1.0000: Pepper Bell Bacterial Spot, Potato Early Blight, Tomato Leaf Mold, Tomato Septoria Leaf Spot, Pepper Bell Healthy, Potato Healthy.

Lowest F1: Tomato Early Blight (0.9766) — minor confusion with Tomato Late Blight. Both diseases produce dark necrotic lesions on tomato foliage and can appear visually similar in early stages, before characteristic ring patterns (Early Blight) or water-soaked margins (Late Blight) are fully developed.

---

## 17. Notebook Cell-by-Cell Guide

| Cell | Title | Description |
|---|---|---|
| 1 | Install Libraries | pip install for PyTorch CUDA 11.8, timm, scikit-learn, seaborn, matplotlib, transformers, Pillow |
| 2 | Mount Google Drive | Connects Colab runtime to Google Drive for persistent file storage |
| 3 | Import Libraries | Imports all modules: os, json, numpy, pandas, matplotlib, PIL, torch, torchvision, timm, sklearn, seaborn, transformers |
| 4 | GPU Detection | torch.device detection — confirms CUDA GPU availability |
| 5 | Download Dataset | Kaggle API authentication, dataset download, and unzip to /content/data/ |
| 6 | Set Dataset Path | Defines data_dir variable pointing to the PlantVillage directory |
| 7 | Dataset Exploration | ImageFolder loading, class count, class distribution bar chart, one-sample-per-class image grid |
| 8 | Preprocessing and Splits | train_transform with augmentation, val/test transform without augmentation, stratified 70/15/15 split using train_test_split, DataLoader creation |
| 9 | Training Function | Defines train_model() with CrossEntropyLoss, Adam optimiser, ReduceLROnPlateau scheduler, early stopping (patience=3), overfitting detection |
| 10 | Train EfficientNet-B0 | Loads pretrained weights, replaces classifier[1] with 15-class Linear layer, calls train_model(), plots training curves |
| 11 | Train ViT-B/16 | timm.create_model with pretrained=True and num_classes=15, calls train_model(), plots training curves |
| 12 | Evaluate Both Models | evaluate() function on test_loader — accuracy, precision, recall, F1, sklearn classification_report, seaborn confusion matrix heatmap |
| 13 | Model Comparison Chart | Pandas DataFrame for results table, grouped bar chart with matplotlib, CSV export |
| 14 | Flan-T5 and AGROVOC | Direct T5ForConditionalGeneration load, run_llm() with beam search, AGROVOC dictionary for all 15 classes |
| 15 | Test Single Image | Full pipeline: load image, preprocess, predict with EfficientNet, generate advisory with Flan-T5, display results |
| 16 | Save Models | torch.save() for both model state_dicts, json.dump() for class_names.json into final_saved_models/ |
| 17 | Export for App | shutil.make_archive() to zip models, Colab files.download() to download to local machine |
| 18 | Launch App | Shows pip install and streamlit run app.py commands |

---

## 18. Known Issues and Fixes

### Issue 1 — Hugging Face pipeline removed in Transformers 4.36+

**Error:** KeyError on text2text-generation or pipeline import failure

**Cause:** The pipeline("text2text-generation") abstraction was removed in Transformers 4.36

**Fix:** Load Flan-T5 directly:
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
```

### Issue 2 — PyTorch 2.x torch.load FutureWarning

**Warning:** FutureWarning: weights_only argument will change default to True in future

**Fix:** Explicitly pass weights_only=True when loading saved model weights:
```python
model.load_state_dict(torch.load("efficientnet_best.pth", weights_only=True))
```

### Issue 3 — Kaggle API not authenticated

**Error:** 401 Unauthorized when downloading dataset in Cell 5

**Fix:** Generate a new API key at kaggle.com > Account > Create New API Token and update the credentials in Cell 5.

### Issue 4 — ViT validation loss spike and early stopping

**Observation:** Validation loss spikes at epoch 5, ReduceLROnPlateau triggers, early stopping triggers at epoch 8.

**Explanation:** ViT-B/16 lacks CNN inductive biases (translational invariance, local connectivity). It requires more data and a lower architecture-specific learning rate (2e-5 with linear warm-up) to stabilise. The shared 1e-4 rate causes instability in early ViT training epochs. This is expected and documented behaviour — 98.5% final accuracy confirms successful convergence despite the earlier stopping.

---

## 19. References

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., . . . Wei, J. (2022). *Scaling instruction-finetuned language models.* arXiv:2210.11416.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale.* arXiv:2010.11929.

FAO. (2021). *The state of food and agriculture 2021: Making agrifood systems more resilient to shocks and stresses.* Food and Agriculture Organisation of the United Nations.

FAO. (2023). *AGROVOC Multilingual Thesaurus.* https://agrovoc.fao.org

Hu, J., Shen, L., Albanie, S., Sun, G., & Wu, E. (2018). *Squeeze-and-excitation networks.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7132-7141.

Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization.* Proceedings of the 3rd International Conference on Learning Representations (ICLR).

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* Proceedings of the 36th International Conference on Machine Learning (ICML). arXiv:1905.11946.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention is all you need.* Advances in Neural Information Processing Systems (NeurIPS), 30.

---

## Author

**K.D. Ruvindu Dulaksha**  
BSc (Hons) Computing — COBSCCOMP4Y241P-018  
NIBM School of Computing and Engineering  
National Institute of Business Management, Sri Lanka  

Demo Video: https://youtu.be/5NuNOEsbK6g

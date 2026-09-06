# Computer-Vision-Project-Multi-Class-Image-Classification

An end-to-end computer vision pipeline that classifies natural-scene photographs into one of six categories: buildings, forest, glacier, mountain, sea, street.

Two models are built and compared:

A CNN trained from scratch (baseline).
A transfer-learning model built on EfficientNetB0 (pretrained on ImageNet), with fine-tuning.
Table of Contents
Problem Statement
Dataset
Project Structure
Setup & Installation
Step-by-Step Walkthrough
Results
How to Run Predictions on New Images
Key Design Decisions
Limitations & Future Improvements

**1. Problem Statement**

Given a photograph of a natural scene, automatically predict which of six categories it belongs to. This is a multi-class, single-label image classification problem — each image belongs to exactly one of six mutually exclusive classes, not multiple at once.

Why this matters: automatic scene classification is a building block for real-world applications such as organizing large photo libraries, tagging satellite/drone imagery, content moderation pipelines, and travel/tourism apps that auto-categorize user photos.

**2. Dataset**

Source: Intel Image Classification (Kaggle)
Classes (6): buildings, forest, glacier, mountain, sea, street
Format: RGB .jpg photographs, variable resolution (predominantly ~150×150)
Splits used in this project: the dataset's own seg_train folder was split 70% train / 15% validation / 15% test (stratified by class), rather than using the Kaggle-provided seg_test folder directly, to have full control over the split ratio. The Kaggle seg_pred folder (unlabeled images) is used later for testing predictions on genuinely unseen images.

To use this project, download the dataset from Kaggle and update the DATASET_PATH variable near the top of the notebook to point to where you've placed it.

**3. Project Structure**

├── notebook.ipynb          # Main Colab/Jupyter notebook — the full pipeline
├── README.md                # This file
├── best_model.keras          # Saved fine-tuned EfficientNetB0 model (after training)
└── outputs/
    └── sample_predictions/    # Example prediction outputs on new images

**4. Setup & Installation**

bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow

A GPU runtime is strongly recommended (e.g. Google Colab with GPU enabled) — training the transfer-learning model on CPU will be considerably slower.

To run:

Open notebook.ipynb in Google Colab or Jupyter.
Update DATASET_PATH to point to your local copy of the Intel Image Classification dataset.
Run all cells top to bottom.
5. Step-by-Step Walkthrough

**Step 1 — Dataset Understanding**

Counts images per class, checks image dimensions, confirms images are RGB (not grayscale), and scans for corrupted/unreadable files using PIL.Image.verify(). Why: you need to know your data before you can preprocess it correctly — for example, whether classes are balanced (they are, roughly ~2,000–2,500 images per class) determines whether you need class weighting or resampling.

**Step 2 — Image EDA (Exploratory Data Analysis)**

Includes a class-distribution bar chart, a grid of sample images per class, image width/height histograms, and pixel-value inspection (confirming the standard 0–255 RGB range). Visual inspection specifically highlights that glacier and mountain images look visually similar (snowy/rocky textures), which is flagged early as a likely source of classification confusion later on.

**Step 3 — Preprocessing**

Each image is:

Decoded and forced to 3 channels (RGB) via tf.image.decode_image(channels=3)
Resized to a fixed size — 150×150 for the baseline CNN, 224×224 for the transfer-learning model (EfficientNetB0's expected input size)
Cast to float32 and normalized to 0–1 for the baseline CNN. The transfer-learning pipeline keeps raw 0–255 values instead, since EfficientNetB0 has its own built-in rescaling layer and expects unnormalized input.
Labels are integer-encoded (buildings=0, forest=1, ...) via a class_to_index mapping, used with sparse_categorical_crossentropy (no one-hot encoding needed).

**Step 4 — Data Splitting**

train_test_split (stratified by label) creates a 70/15/15 train/validation/test split. Why data leakage must be avoided: if test images are seen during training, or used repeatedly to pick the "best" model, the reported accuracy stops reflecting how the model will perform on truly new data. The test set here is split off first and never touched until final evaluation — no augmentation, no model-selection decisions are based on it.

**Step 5 — Data Augmentation**

Applied only to the training set, using tf.keras.layers:

Augmentation	Why it's appropriate for natural scenes
Random horizontal flip	A mirrored forest/mountain/sea is still the same scene category
Small random rotation (±5%)	Simulates minor camera-angle variation without creating unrealistic upside-down scenes
Random zoom (±10%)	Simulates photos taken from different distances
Random translation (±10%)	Prevents the model from relying on exact object position in the frame
Random contrast (±10%)	Simulates different lighting/weather conditions

Augmentation is applied inside the tf.data pipeline with training=True explicitly set, after batching, and is applied via two independent augmentation stacks — one built for 150×150 inputs (baseline) and a separate one for 224×224 inputs (transfer learning) — to avoid a shape-mismatch error from reusing one augmentation layer across two different input resolutions.

**Step 6 — Baseline CNN (built from scratch)**

Architecture: 4 convolutional blocks (32→32→64→64→128→256 filters), each with Conv2D + BatchNormalization, MaxPooling2D between blocks, followed by GlobalAveragePooling2D, two Dropout-regularized dense layers, and a final 6-way softmax output.

Optimizer: Adam, learning rate 3e-4 — chosen after an initial run at 1e-3 showed unstable, oscillating validation accuracy; the lower rate combined with ReduceLROnPlateau (halving the LR when validation loss stalls) produced smooth, stable convergence.
Regularization: BatchNormalization after every conv layer, Dropout (0.2–0.4) at multiple points, and EarlyStopping (patience=6, restoring best weights) to prevent overfitting.

**Step 7 — Training Curves**

Accuracy and loss are plotted for training vs. validation across epochs. By the final epoch, training accuracy reaches ~0.88 while validation accuracy reaches ~0.84 — a modest, healthy gap indicating the model generalizes reasonably well without severe overfitting.

**Step 8 — Model Evaluation**

Evaluated on the untouched test set with accuracy, precision, recall, F1-score (overall and per-class), and a confusion matrix. Reporting accuracy alone is avoided since it can hide poor performance on individual classes.

**Step 9 — Error Analysis**

Misclassified test images are displayed with their true label, predicted label, and model confidence. The confusion matrix is scanned for the single most common misclassification pattern, and a full ranked table of all confusion pairs is produced — confirming the glacier↔mountain confusion flagged during EDA is indeed the model's biggest source of error, consistent with their visual similarity.

**Step 10 — Transfer Learning**

A second model is built on EfficientNetB0 (ImageNet weights, top layers excluded). The pretrained base is frozen, and a custom classification head (GlobalAveragePooling2D → Dropout → Dense(128) → Dropout → Dense(6, softmax)) is added on top. For fine-tuning, the base is unfrozen and all but the top 20 layers are re-frozen, then the whole model is recompiled with a much smaller learning rate (1e-5) so that the pretrained weights are only nudged slightly rather than overwritten.

**Step 11 — Model Comparison**

A comparison table (input size, trainable parameters, validation/test accuracy, F1-score, training time) is built for both models — see Results below.

**Step 12 — Prediction on New Images**

A reusable predict_image() function loads an arbitrary image file, preprocesses it to match whichever model it's given (handling the different rescaling requirements of the baseline CNN vs. the transfer-learning model), and returns the predicted class, confidence score, and a displayed copy of the image. Tested on 5 images from the Kaggle-provided seg_pred folder — images the model never saw during training, validation, or testing.

**6. Results**
Model	Input Size	Trainable Params	Val Accuracy	Test Accuracy	F1-Score (weighted)	Training Time
Baseline CNN (from scratch)	150×150	469,414	0.837	0.848	0.848	16.4 min
EfficientNetB0 (fine-tuned)	224×224	1,515,702	0.923	0.917	0.917	17.2 min

Per-class performance (fine-tuned EfficientNetB0):

Class	Precision	Recall	F1-score
buildings	0.96	0.88	0.92
forest	0.98	0.99	0.98
glacier	0.83	0.88	0.85
mountain	0.91	0.84	0.87
sea	0.95	0.96	0.96
street	0.89	0.96	0.93

Which model would be chosen for production? The fine-tuned EfficientNetB0 model, given its clearly higher accuracy and F1-score for only a modest increase in training time. The trade-off is a larger model to serve at inference time and a 224×224 input requirement, versus the baseline's smaller, faster-to-serve 150×150 model — but the ~7-point accuracy gain outweighs that cost for most deployment scenarios.

Hardest classes to classify: glacier and mountain, consistent with the visual similarity flagged during EDA (both feature rocky/snowy textures and similar color palettes). forest is the easiest class to classify (F1 = 0.98), likely due to its distinctive green color palette and texture.

**7. How to Run Predictions on New Images**
python
predicted_class, confidence, probabilities = predict_image(
    filepath="path/to/your/image.jpg",
    model=transfer_model,      # the fine-tuned EfficientNetB0 model
    img_size=(224, 224),
    class_names=class_names,
    rescale=False               # EfficientNetB0 expects raw 0-255 input
)
**8.Key Design Decisions**
Manual 70/15/15 split instead of the dataset's provided seg_train/seg_test folders, to control the exact split ratio and keep the split stratified by class.
Two separate augmentation pipelines (one per image resolution) rather than one shared pipeline, to avoid input-shape conflicts between the 150×150 baseline model and the 224×224 transfer-learning model.
Augmentation applied inside the tf.data pipeline, not inside the model itself — this keeps the training=True behavior explicit in code rather than relying on Keras to infer it, and avoids the augmentation layer's input shape becoming permanently locked to whichever model first used it.
Lower learning rate (3e-4) + ReduceLROnPlateau for the baseline CNN, after an initial run at 1e-3 produced unstable, oscillating validation accuracy across epochs.
9. Limitations & Future Improvements
Only the top 20 layers of EfficientNetB0 were unfrozen during fine-tuning; unfreezing more layers (with careful learning-rate scheduling) could potentially improve accuracy further.
No explainability technique (e.g. Grad-CAM) is currently included — adding one would help visualize which parts of glacier/mountain images the model finds ambiguous.
No deployment interface (e.g. Streamlit) is included yet.
More training data, or targeted data augmentation for the glacier/mountain classes specifically, could help close the remaining gap in their F1-scores.

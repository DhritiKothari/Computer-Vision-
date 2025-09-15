 Advanced Image Classifier for CIFAR-100 using Transfer Learning

This project builds a high-performance image classifier using Transfer Learning. It leverages the pre-trained MobileNetV2 model to classify images from the more challenging CIFAR-100 dataset.


 1. Dataset

The project uses the CIFAR-100 dataset.
-   Challenge: This dataset contains 100 classes, making it significantly more difficult than CIFAR-10.
-   Content: 60,000 color images (32x32 pixels), with 600 images per class.
-   Split: 50,000 for training, 10,000 for testing.

 2. Preprocessing & Data Augmentation

To prepare the data for the pre-trained model and improve its robustness, several key steps were taken:

-   Image Resizing: CIFAR-100 images (32x32) were upsampled to 96x96, a more suitable size for the MobileNetV2 architecture.
-   Model-Specific Preprocessing: Applied the `preprocess_input` function specific to MobileNetV2, which normalizes images according to how the model was originally trained.
-   Data Augmentation: During training, random transformations (`RandomFlip`, `RandomRotation`) were applied to the images. This artificially expands the dataset, reduces overfitting, and helps the model generalize better to unseen data.

3. Workflow: Transfer Learning & Fine-Tuning

This project employs a two-phase training strategy to achieve the best possible performance:

Phase 1: Feature Extraction
1.  The MobileNetV2 base model, pre-trained on ImageNet, was loaded without its top classification layer.
2.  The entire base model was frozen so its learned weights would not be updated.
3.  A new classifier head (a `GlobalAveragePooling2D` layer and a `Dense` layer with 100 outputs) was added on top.
4.  The model was trained for 10 epochs. In this phase, only the weights of the new classifier head were trained.

#### Phase 2: Fine-Tuning
1.  The base model was **unfrozen** to make its weights trainable.
2.  To avoid destroying the learned features, only the top layers of the base model (from layer 100 onwards) were set to be trainable.
3.  The model was re-compiled with a very **low learning rate** (`0.00001`).
4.  Training continued for another 10 epochs, allowing the model to make small adjustments to its pre-trained weights to better adapt to the CIFAR-100 dataset.

4. Results

This advanced approach yielded a significant performance improvement.

-   Final Test Accuracy: 71.35% (Update this with your final accuracy!)

The training plot clearly shows the two phases. The model learns rapidly during the initial phase and then steadily improves during fine-tuning.

![Advanced Training History](advanced_training_history.png)

 5. How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the script:
    ```bash
    python main.py
    ```
The script will download the dataset and pre-trained weights (on the first run), train the model in two phases, save the training plot, and print the final test accuracy.

# Human-Emotion-Detection
Tell about emotion of a person (Angry, Happy, Sad)

# Running Prototype
Link : https://huggingface.co/spaces/deep42/Human-Emotion-Detection

# Dataset I Used 
- For training of model i used dataset available kaggle [here]()
  
![download](https://github.com/deep-gtm/Human-Emotion-Detection/assets/70434931/e7dffc98-a007-47d1-b702-66be2d08012f)

# Prediction Resutls :

![download (1)](https://github.com/deep-gtm/Human-Emotion-Detection/assets/70434931/905e5b41-f0ba-4baa-9468-1aef1134ebef)

# Basic Things I Learned In This Project :
1. We can use convolution neural network for image classification purpose.
2. So a basic image classification model consist of two part
3. First part is made up of convolution layers that extract features from a image (low and high level features)
4. Second part is made up of connected layers that do classification.

# Tools And Technology
1. There are different convolution neural network which helped in computer vision tasks
  1. Alex Net
     - AlexNet is a pioneering deep neural network architecture for computer vision.
     - It has convolutional layers, ReLU activations, max-pooling layers, and fully connected layers.
     - It introduced techniques like local response normalization and GPU training.
  2. VGG
     - It has multiple convolutional layers with small filters, followed by max-pooling layers.
     - VGG16 and VGG19 are popular variants with 16 and 19 layers, respectively.
     - VGG is known for its simplicity and strong feature extraction capabilities.
     - However, it requires more computational resources due to its depth.
  3. ResNet
     - It introduced skip connections to tackle the vanishing gradient problem.
     - These connections allow information to bypass certain layers, enabling the training of very deep networks.
     - ResNet has a "bottleneck" design, using 1x1 convolutions to reduce computational complexity.
  4. Mobile Net
     - MobileNet is a lightweight neural network architecture designed specifically for mobile and embedded devices with limited computational resources.
     - It employs depth-wise separable convolutions, which split the standard convolutional operation into separate depth-wise and point-wise convolutions, reducing the computational complexity.
     - MobileNet achieves a good balance between model size and accuracy, making it well-suited for applications that require efficient inference on resource-constrained devices.
  5. Efficient Net
     - EfficientNet aims to strike a balance between model size, computational efficiency, and accuracy.
     - It achieves this balance through a compound scaling method that uniformly scales network depth, width, and resolution. EfficientNet uses a mobile inverted bottleneck structure inspired by MobileNetV2, which employs depth-wise separable convolutions to reduce computations.
     - By finding the optimal scaling factors, EfficientNet provides efficient models suitable for resource-constrained devices without sacrificing performance.

2. Transfer Learning :
     - Transfer learning is a machine learning technique where knowledge gained from training one model on a specific task is transferred and applied to another related task.
     - Instead of training a model from scratch, transfer learning utilizes pre-trained models that have been trained on large-scale datasets, typically for generic tasks like image classification.
     - The pre-trained model's learned representations and knowledge are leveraged as a starting point for the new task, allowing for faster and more effective training with limited data.
     - Transfer learning enables the transfer of general knowledge across tasks, benefiting from the features and patterns learned by the pre-trained model.

3. Fine Tunning:
     - Fine-tuning is a transfer learning technique that adapts a pre-trained model to a new task.
     - It involves reusing the learned representations of the pre-trained model and training new layers on a task-specific dataset.
     - Fine-tuning allows for faster convergence and better performance, particularly when data is limited or the tasks are similar.
4. Visualizing Intermediate Layer.
5. GradCam Method :
     - Grad-CAM is a visualization technique used to understand deep neural networks.
     - It highlights important regions in an input image that contribute to a network's prediction for a specific class.
     - It computes gradients, weights activation maps, and overlays them on the image to visualize areas of focus.
  
6. ViT (Vision Transformer):
     - Vision Transformer (ViT) is a deep learning architecture that applies the transformer model, originally developed for natural language processing (NLP), to computer vision tasks.
     - It replaces the traditional convolutional layers found in popular vision models with self-attention mechanisms used in transformers. ViT divides an image into patches and feeds them through a transformer encoder, enabling global interactions among patches.
     - It has shown promising performance on various image classification tasks and has gained attention for its ability to handle long-range dependencies in images.
     - In this project i create a vit model using TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
7. Grado :
     - Gradio is a Python library that helps you build and deploy web interfaces for machine learning models.
     - It simplifies the creation of interactive user interfaces, allowing users to input data and see real-time predictions or results from the model.
     - It supports various machine learning frameworks and is popular for creating interactive demonstrations of models.
8. Huggingface and Huggingface spaces
9. ONNX :
      - ONNX (Open Neural Network Exchange) is an open-source format that enables interoperability between different deep learning frameworks.
      - It allows you to train and deploy machine learning models using one framework and seamlessly transfer them to another framework for inference.
      - ONNX provides a standardized way to represent models, including their architecture and trained parameters, allowing for efficient model sharing and deployment across platforms and devices.
      - t supports a wide range of frameworks, including TensorFlow, PyTorch, and more, making it easier to work with models in a heterogeneous deep learning ecosystem.

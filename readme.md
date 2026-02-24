Handwritten Digit Recognition Using Machine Learning
📌 Project Overview
This project implements an end-to-end applied machine learning system to recognize handwritten digits (0–9).
 The system is trained on the MNIST dataset and demonstrates the full ML pipeline including:
Dataset collection


Data preprocessing


Feature preparation


Model training


Prediction


Performance evaluation


The goal is to build a supervised classification model capable of accurately identifying digits from grayscale images.

🎯 Objective
To design and implement a complete machine learning system that:
Processes raw image data


Learns patterns from labeled digits


Predicts unseen handwritten digits


Evaluates performance using appropriate classification metrics



📊 Dataset Description
Dataset Used: MNIST
Total Images: 70,000


Image Size: 28 × 28 pixels


Color Type: Grayscale


Number of Classes: 10 (Digits 0–9)


Each image is converted into a feature vector of 784 pixels (28 × 28 = 784).

🔄 System Workflow
The system follows the standard machine learning pipeline:

1️⃣ Data Collection
The MNIST dataset is loaded using Scikit-learn's dataset API.
Each sample contains:
784 pixel intensity values


Corresponding digit label



2️⃣ Data Preprocessing
✔ Normalization
Pixel values range from 0 to 255.
 They are scaled to 0–1 using:
X = X / 255.0
Why?
Improves model convergence


Prevents large values from dominating


Makes training more stable



✔ Train-Test Split
The dataset is split into:
80% Training Data


20% Testing Data


Purpose:
Training set → Model learns patterns


Testing set → Evaluate real-world performance



3️⃣ Feature Preparation
Each 28×28 image is flattened into a 1D feature vector of 784 values.
Example:
[Pixel1, Pixel2, Pixel3, ..., Pixel784]
These pixel intensities act as numerical features for the model.

4️⃣ Model Development
Algorithm Used: Logistic Regression
Type: Multiclass Classification
 Approach: One-vs-Rest (OvR)
Why Logistic Regression?
Suitable for classification problems


Performs well on linearly separable data


Computationally efficient


Good baseline model for image classification



Mathematical Concept
The model computes:
Probability = Softmax(WX + b)
Where:
W = weight matrix


X = input features (pixels)


b = bias


Softmax = converts outputs into probabilities


The class with the highest probability is predicted.

5️⃣ Model Training
The model is trained using the training dataset.
During training:
The algorithm adjusts weights


Minimizes classification error


Optimizes using gradient descent



6️⃣ Prediction
The trained model predicts digits for unseen test images.
Each prediction assigns one of 10 classes (0–9).

📈 Performance Evaluation
The model is evaluated using multiple metrics:

✅ Accuracy
Measures overall correctness.
Accuracy = (Correct Predictions) / (Total Predictions)
Typical Result: ~93–95%

✅ Precision
Out of predicted digits, how many were correct?

✅ Recall
Out of actual digits, how many were correctly identified?

✅ F1-Score
Harmonic mean of Precision and Recall.
Used when class balance matters.

✅ Confusion Matrix
Displays classification performance in detail.
Shows:
Correct classifications on diagonal


Misclassifications off-diagonal


Helps identify which digits are confused (e.g., 5 vs 8).

🛠 Technologies Used
Python


NumPy


Matplotlib


Scikit-learn


Visual Studio Code



🧠 Challenges Faced
Handling large dataset efficiently


Ensuring proper normalization


Avoiding convergence warnings (solved using max_iter parameter)



🚀 Future Improvements
Implement Convolutional Neural Network (CNN)


Improve accuracy beyond 98%


Add GUI for drawing digits


Deploy as web application


Implement real-time digit recognition using camera



📌 Conclusion
This project successfully demonstrates an end-to-end machine learning workflow from raw image data to classification output.
It showcases understanding of:
Supervised learning


Data preprocessing


Feature engineering


Model training


Performance evaluation


Model interpretation


The model achieves strong classification accuracy and provides a solid foundation for advanced computer vision applications.


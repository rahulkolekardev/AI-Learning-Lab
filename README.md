# 🧪 AI Learning Lab - Interactive Supervised Algorithm Playground 🚀

Welcome to the AI Learning Lab! This interactive web application allows you to explore, visualize, and understand the workings of various supervised machine learning algorithms right in your browser. Built with vanilla JavaScript, HTML, and CSS, it provides a hands-on experience with data generation, hyperparameter tuning, model training, and performance evaluation.

**✨ Live Demo:** [https://rahulkolekardev.github.io/AI-Learning-Lab/](https://rahulkolekardev.github.io/AI-Learning-Lab/)

**📂 Repository:** [https://github.com/rahulkolekardev/AI-Learning-Lab](https://github.com/rahulkolekardev/AI-Learning-Lab)

## 🖼️ Application Preview

![AI Learning Lab Screenshot](https://github.com/rahulkolekardev/AI-Learning-Lab/blob/main/Screenshot.png?raw=true)
*(Screenshot of the AI Learning Lab in action!)*

## 🌟 Features

*   **🤖 Interactive Algorithm Selection:** Choose from a suite of classic supervised learning algorithms:
    *   **📈 Linear Regression (Advanced):** With detailed metrics (MSE, RMSE, MAE, R²).
    *   **🤝 k-Nearest Neighbors (k-NN Classifier):** Simple instance-based learning.
    *   **🎯 Logistic Regression (Classifier):** For binary classification tasks, with L2 Regularization.
    *   **🌳 Decision Tree (Classifier):** Visualizes "blocky" decision boundaries.
    *   **📧 Naive Bayes (Gaussian Classifier):** Probabilistic classification.
    *   **🎯hyperplane Support Vector Machine (SVM Classifier):** With Linear and a simplified RBF kernel concept.
    *   **🌲 Random Forest (Classifier):** Ensemble of decision trees (simplified).
*   **📊 Dynamic Dataset Generation:**
    *   Create various 2D datasets for both classification and regression tasks:
        *   Gaussian Blobs 🔵🔴
        *   Concentric Circles 🎯
        *   XOR-like patterns 🏁
        *   Linear Trends 📉
        *   Sine Waves 〰️
    *   Control the number of samples and noise level.
*   **⚙️ Hyperparameter Tuning:**
    *   Adjust key hyperparameters specific to each algorithm in real-time.
    *   Conditional display of parameters (e.g., for SVM kernels).
*   **💡 Live Training & Visualization:**
    *   Train models instantly with the click of a button.
    *   **For Classifiers:** Visualize learned decision boundaries, showing how the model separates the data space.
    *   **For Regressors:** See the fitted regression line.
    *   Data points are plotted for clear context.
*   **📈 Comprehensive Performance Metrics:**
    *   **Classification:** Accuracy.
    *   **Regression:** Mean SquaredError (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
*   **🎨 "AI Lab" Dark Theme UI:**
    *   A modern, futuristic, dark-themed interface designed for an engaging learning experience.
    *   Responsive design for various screen sizes.
*   **🧩 Modular Code Structure:**
    *   Organized with separate JavaScript files for each algorithm and a central `main.js` for application logic, promoting maintainability and extensibility.
*   **🎓 Educational Focus:**
    *   Each algorithm panel includes a brief description of its principles.

## 🛠️ Tech Stack

*   **HTML5:** Structure of the application.
*   **CSS3:** Styling, including the dark theme and responsive layout.
*   **Vanilla JavaScript (ES6+):** All core logic, algorithm implementations, and DOM manipulation. No external JS libraries or frameworks are used for the core ML parts.

## 📁 Project Structure

```
AI-Learning-Lab/
├── index.html           # Main application page
├── style.css            # All CSS styles
├── js/
│   ├── main.js          # Core application logic, UI management, event handling
│   └── algorithms/      # Folder for individual algorithm implementations
│       ├── linearRegression.js
│       ├── knn.js
│       ├── logisticRegression.js
│       ├── decisionTree.js
│       ├── naiveBayes.js
│       ├── svm.js
│       └── randomForest.js
└── README.md            # This file
```

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rahulkolekardev/AI-Learning-Lab.git
    cd AI-Learning-Lab
    ```
2.  **Open `index.html` in your web browser:**
    *   Simply double-click the `index.html` file, or open it using your browser's "File > Open" menu.
    *   A modern browser (Chrome, Firefox, Edge, Safari) is recommended. 🌐

## 🎮 How to Use the Playground

1.  **Visit the Live Demo:** [https://rahulkolekardev.github.io/AI-Learning-Lab/](https://rahulkolekardev.github.io/AI-Learning-Lab/)
2.  **Select an Algorithm:** Choose the supervised learning algorithm you want to explore from the "Select Protocol" dropdown.
3.  **Configure Dataset:**
    *   Select a `Dataset Matrix` type appropriate for the chosen algorithm (classification or regression).
    *   Adjust `Sample Density` (number of data points) and `Signal Noise`.
    *   Click `Generate Data Matrix`. You'll see the data points plotted.
4.  **Tune Hyperparameters:**
    *   The "CALIBRATION" panel will show hyperparameters specific to the selected algorithm. Adjust them as desired.
5.  **Train the Model:**
    *   Click the `Initiate Training` button.
    *   The model will be trained on the current dataset and hyperparameters.
6.  **Observe Results:**
    *   **Visualization Grid:** The learned decision boundary (for classifiers) or regression line (for regressors) will be overlaid on the data points.
    *   **Performance Analysis:** Relevant metrics like Accuracy, MSE, R², etc., will be displayed.
    *   **Algorithm Info:** Read a brief description of how the selected algorithm works.
7.  **Experiment!** 🎉
    *   Try different algorithms with various datasets and hyperparameter settings.
    *   Click `System Reset` to clear the current model and generate a fresh dataset with current settings.
    *   Click `Generate Data Matrix` to get new data points while keeping the algorithm and hyperparameters.

## 🔮 Future Enhancements (Potential Ideas)

*   **More Algorithms:**
    *   Support Vector Regressor (SVR).
    *   Gradient Boosting Trees (simplified).
    *   🧠 Neural Network (integrated as another option).
*   **Advanced Hyperparameter Options:**
    *   More kernel options for SVM.
    *   Different impurity measures for Decision Trees (e.g., Entropy).
    *   Feature sub-sampling for Random Forest.
*   **Improved Visualizations:**
    *   📈 Plotting cost function / learning curves for iterative algorithms.
    *   🌳 Visualizing the structure of Decision Trees.
    *   Show support vectors for SVM.
    *   Residual plots for regression.
*   **Data Import/Export:** Allow users to upload their own simple CSV datasets. 📄
*   **Step-by-Step Training:** For iterative algorithms, allow users to step through training iterations. 🚶‍♂️
*   **Cross-Validation Simulation:** To demonstrate hyperparameter tuning strategies. 🔍
*   **More Detailed Metric Breakdowns:** Confusion matrices, precision-recall curves for classifiers. 📜

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/rahulkolekardev/AI-Learning-Lab). Let's make this even better together! 🧑‍💻

---


# AI Learning Lab: Interactive ML Algorithm Playground

## 🚀 About The Project

AI Learning Lab is a browser-based, interactive playground designed to help users visualize and understand the workings of various fundamental supervised machine learning algorithms. Built entirely with vanilla JavaScript, HTML, and CSS, this application provides a hands-on experience without requiring any complex setup or external libraries for the core algorithm implementations.

Users can:
*   Select from a range of supervised learning algorithms (k-NN, Linear Regression, Logistic Regression, Decision Trees).
*   Generate diverse 2D datasets (Gaussian blobs, concentric circles, XOR-like patterns, linear trends, sine waves) with configurable sample sizes and noise levels.
*   Tune algorithm-specific hyperparameters in real-time.
*   Initiate model training directly in the browser.
*   Visualize the learned model:
    *   **Classifiers:** See the decision boundaries that separate classes.
    *   **Regressors:** Observe the fitted regression line.
*   Analyze model performance through key metrics like Accuracy and Mean Squared Error (MSE).
*   Gain insights from brief explanations about each algorithm.

The application features a sleek, dark-themed "AI/futuristic" user interface to enhance the learning experience.

## ✨ Features

*   **Interactive Algorithm Exploration:**
    *   k-Nearest Neighbors (Classifier)
    *   Linear Regression
    *   Logistic Regression (Classifier)
    *   Decision Tree (Classifier - Gini Impurity based)
*   **Dynamic Dataset Generation:**
    *   Classification: Gaussian Blobs, Concentric Circles, XOR-like.
    *   Regression: Linear Trend, Sine Wave.
    *   Adjustable number of samples and noise levels.
*   **Real-time Hyperparameter Tuning:** Modify parameters like `k` for k-NN, learning rate & iterations for regression models, max depth for Decision Trees, and see immediate effects (after retraining).
*   **In-Browser Training:** Models are trained using JavaScript implementations of their respective algorithms (e.g., Gradient Descent for regressions).
*   **Rich Visualizations:**
    *   Live plotting of generated datasets.
    *   Dynamic decision boundary rendering for classifiers.
    *   Regression line plotting for regressors.
*   **Performance Metrics:**
    *   Accuracy for classification tasks.
    *   Mean Squared Error (MSE) for regression tasks.
*   **Informative UI:**
    *   Sleek, dark-themed "AI Lab" aesthetic.
    *   Tooltips and descriptions for algorithms and parameters.
*   **Pure Vanilla JS, HTML, CSS:** No external ML libraries for core algorithms, making it lightweight and a great learning tool for understanding implementations from scratch.

## 🛠️ Built With

*   Vanilla JavaScript (ES6+)
*   HTML5
*   CSS3

## 🚀 Getting Started

1.  Clone the repository:
    ```sh
    git clone https://YOUR_USERNAME/ai-learning-lab.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd ai-learning-lab
    ```
3.  Open the `index.html` (or the main HTML file, e.g., `ai_lab_playground.html`) file in your favorite modern web browser.

No installation or build steps are required!

## 📖 Usage

1.  **Select an Algorithm:** Choose from the "SYSTEM CORE // Algorithm" dropdown.
2.  **Configure Dataset:**
    *   Select a "Dataset Matrix" type.
    *   Adjust "Sample Density" and "Signal Noise."
    *   Click "Generate Data Matrix."
3.  **Calibrate Hyperparameters:** Modify the parameters specific to the selected algorithm in the "CALIBRATION" panel.
4.  **Initiate Training:** Click the "Initiate Training" button.
5.  **Observe & Analyze:**
    *   Watch the "VISUALIZATION GRID" update with data points and the model's learned representation (decision boundary or regression line).
    *   Check the "PERFORMANCE ANALYSIS" panel for metrics like Accuracy or MSE.
    *   Read about the algorithm in the "About" section.
6.  **Experiment:** Try different algorithms, datasets, and hyperparameter combinations to see how they affect model behavior and performance. Use "System Reset" to start fresh.

## 🛣️ Future Roadmap

*   [ ] Add more supervised learning algorithms (e.g., Naive Bayes, SVM with basic kernels, Random Forest).
*   [ ] Implement regression versions for k-NN and Decision Trees.
*   [ ] Introduce more complex datasets or an option to upload simple CSV data.
*   [ ] Enhance visualizations (e.g., plotting loss curves for iterative algorithms, showing decision tree structure).
*   [ ] Add more detailed performance metrics (e.g., confusion matrix, precision, recall, F1-score, R-squared).
*   [ ] Implement cross-validation concepts (simplified).
*   [ ] Allow step-by-step execution for iterative algorithms to visualize learning.
*   [ ] Explore Web Workers for offloading computationally intensive tasks (like decision boundary rendering for complex models).

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information. (You would need to create a LICENSE file with MIT license text).

## 🙏 Acknowledgements

*   Inspired by various online interactive machine learning demonstrations.
*   Fonts from Google Fonts (Orbitron, Roboto).

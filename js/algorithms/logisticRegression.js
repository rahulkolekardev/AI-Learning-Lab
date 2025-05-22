// js/algorithms/logisticRegression.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};

    const logisticRegressionConfig = {
        name: 'Logistic Regression',
        type: 'classifier',
        description: `Logistic Regression is a statistical model used for binary classification. 
                      It models the probability of a dichotomous outcome (e.g., 0 or 1) using a logistic (sigmoid) function 
                      applied to a linear combination of input features. The model is trained using Gradient Descent 
                      to minimize the Log Loss (Binary Cross-Entropy) cost function. 
                      It's effective for linearly separable data or problems where a linear decision boundary is appropriate.`,
        params: [
            { 
                id: 'logRegLearningRate', 
                label: 'Learning Rate (α)', 
                type: 'number', 
                min: 0.0001, 
                max: 1.0, 
                step: 0.0001, 
                default: 0.05 // Adjusted default for potentially better convergence
            },
            { 
                id: 'logRegIterations', 
                label: 'Training Iterations', 
                type: 'number', 
                min: 100, 
                max: 5000, // Increased max for more complex datasets
                step: 50, 
                default: 1000 
            },
            {
                id: 'logRegLambda',
                label: 'L2 Regularization (λ)',
                type: 'number',
                min: 0,
                max: 0.1,
                step: 0.001,
                default: 0.001 // Small default regularization
            }
        ],
        train: trainLogisticRegression,
        predict: predictLogisticRegression,
        isIterative: true 
    };

    function sigmoid(z) {
        // Clip z to prevent overflow/underflow in Math.exp
        const Z_MAX = 30; // Values beyond this make sigmoid very close to 0 or 1
        const Z_MIN = -30;
        if (z > Z_MAX) return 1;
        if (z < Z_MIN) return 0;
        return 1 / (1 + Math.exp(-z));
    }

    function trainLogisticRegression(dataset, hyperparameters) {
        if (!dataset || dataset.length === 0) {
            console.warn("Logistic Regression Training: Dataset is empty.");
            return null;
        }
        const numFeatures = dataset[0].inputs.length;
        // Initialize weights and bias with small random values or zeros
        let weights = Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.02); 
        let bias = (Math.random() - 0.5) * 0.02;

        const learningRate = hyperparameters.logRegLearningRate;
        const iterations = hyperparameters.logRegIterations;
        const lambda = hyperparameters.logRegLambda || 0; // L2 regularization strength

        const N = dataset.length;
        const costHistory = []; // For potential future plotting

        for (let iter = 0; iter < iterations; iter++) {
            let gradientWeights = Array(numFeatures).fill(0);
            let gradientBias = 0;
            let currentEpochCost = 0; // Log Loss

            dataset.forEach(sample => {
                // Linear combination: z = w*x + b
                let z = bias;
                sample.inputs.forEach((inputVal, featureIndex) => {
                    z += weights[featureIndex] * inputVal;
                });

                // Activation: prediction_proba = sigmoid(z)
                const predictionProba = sigmoid(z);

                // Error for gradient calculation: (prediction_proba - target)
                const error = predictionProba - sample.target; // sample.target is 0 or 1

                // Accumulate gradients
                gradientBias += error;
                sample.inputs.forEach((inputVal, featureIndex) => {
                    gradientWeights[featureIndex] += error * inputVal;
                });

                // Calculate Log Loss (Binary Cross-Entropy) for this sample
                // Cost = -[y*log(p) + (1-y)*log(1-p)]
                // Add a small epsilon to prevent log(0)
                const epsilon = 1e-9;
                currentEpochCost += - (sample.target * Math.log(predictionProba + epsilon) + 
                                     (1 - sample.target) * Math.log(1 - predictionProba + epsilon));
            });
            
            currentEpochCost /= N; // Average cost over the dataset
            // Add L2 regularization cost component (optional, for monitoring)
            // if (lambda > 0) {
            //     let sumSqWeights = 0;
            //     weights.forEach(w => sumSqWeights += w*w);
            //     currentEpochCost += (lambda / (2 * N)) * sumSqWeights;
            // }
            costHistory.push(currentEpochCost);

            // Update bias
            bias -= learningRate * (gradientBias / N);
            
            // Update weights with L2 regularization
            weights.forEach((weight, featureIndex) => {
                const regularizationTerm = (lambda / N) * weight; // L2 penalty gradient term
                weights[featureIndex] -= learningRate * ((gradientWeights[featureIndex] / N) + regularizationTerm);
            });

            if (iterations > 200 && iter % Math.floor(iterations / 10) === 0) {
                 console.log(`LogReg Iter ${iter}: Cost=${currentEpochCost.toFixed(6)}`);
            }
        }
        
        // console.log("Final Logistic Regression Model:", { weights, bias, costHistory });
        return { 
            weights, 
            bias, 
            type: 'classifier', 
            trainingSummary: { finalCost: costHistory[costHistory.length-1], iterationsRun: iterations } 
        };
    }

    function predictLogisticRegression(pointInputs, model) {
        if (!model || !model.weights || typeof model.bias === 'undefined') {
            // console.warn("Logistic Regression Prediction: Model is invalid.");
            return -1; // Or throw an error
        }

        let z = model.bias;
        pointInputs.forEach((inputVal, featureIndex) => {
            if (model.weights[featureIndex] !== undefined) {
                z += model.weights[featureIndex] * inputVal;
            }
        });

        const probability = sigmoid(z);
        return probability > 0.5 ? 1 : 0; // Threshold probability to get class label
    }

    // Expose the configuration to the global scope
    window.ALGORITHMS.logisticRegression = logisticRegressionConfig;

})();
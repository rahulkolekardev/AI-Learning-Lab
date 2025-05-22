// js/algorithms/linearRegression.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};

    const linearRegressionConfig = {
        name: 'Linear Regression',
        type: 'regressor',
        description: 'Advanced Linear Regression: Models linear relationships using Gradient Descent. Evaluated with MSE, RMSE, MAE, and R-squared.',
        params: [
            { 
                id: 'lrLearningRate', label: 'Learning Rate (α)', type: 'number', 
                min: 0.00001, max: 0.5, step: 0.00001, default: 0.05 
            },
            { 
                id: 'lrIterations', label: 'Training Iterations', type: 'number', 
                min: 50, max: 2000, step: 10, default: 300 
            },
            // Example of adding a regularization parameter (conceptual, not fully implemented in train yet)
            // {
            //     id: 'lrL2Lambda', label: 'L2 Regularization (λ)', type: 'number',
            //     min: 0, max: 0.1, step: 0.001, default: 0
            // }
        ],
        train: trainLinearRegression,
        predict: predictLinearRegression,
        isIterative: true
    };

    function trainLinearRegression(dataset, hyperparameters) {
        const numFeatures = dataset[0].inputs.length;
        let weights = Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1); // Small random weights
        let bias = (Math.random() - 0.5) * 0.1;

        const learningRate = hyperparameters.lrLearningRate;
        const iterations = hyperparameters.lrIterations;
        // const lambda = hyperparameters.lrL2Lambda || 0; // For L2 regularization (Ridge)

        const costHistory = []; // To potentially plot cost function later

        for (let iter = 0; iter < iterations; iter++) {
            let gradientWeights = Array(numFeatures).fill(0);
            let gradientBias = 0;
            let currentEpochMse = 0;

            dataset.forEach(sample => {
                let prediction = bias;
                sample.inputs.forEach((inputVal, featureIndex) => {
                    prediction += weights[featureIndex] * inputVal;
                });

                const error = prediction - sample.target;
                currentEpochMse += error * error;

                gradientBias += error;
                sample.inputs.forEach((inputVal, featureIndex) => {
                    gradientWeights[featureIndex] += error * inputVal;
                });
            });
            
            currentEpochMse /= dataset.length;
            costHistory.push(currentEpochMse);

            // Update bias
            bias -= learningRate * (gradientBias / dataset.length);
            
            // Update weights (with L2 regularization if lambda > 0)
            weights.forEach((weight, featureIndex) => {
                const regularizationTerm = 0; // (lambda / dataset.length) * weight; // L2 penalty
                weights[featureIndex] -= learningRate * ((gradientWeights[featureIndex] / dataset.length) + regularizationTerm);
            });

            // Optional: Log progress for debugging
            if (iterations > 100 && iter % Math.floor(iterations / 10) === 0) {
                 console.log(`LinReg Iter ${iter}: MSE=${currentEpochMse.toFixed(6)}`);
            }
        }
        
        // console.log("Final Linear Regression Model:", { weights, bias });
        return { weights, bias, type: 'regressor', trainingSummary: { finalMse: costHistory[costHistory.length-1], iterationsRun: iterations } };
    }

    function predictLinearRegression(pointInputs, model) {
        let prediction = model.bias;
        pointInputs.forEach((inputVal, featureIndex) => {
            // Ensure model.weights[featureIndex] exists, especially if numFeatures changed
            if (model.weights && model.weights[featureIndex] !== undefined) {
                prediction += model.weights[featureIndex] * inputVal;
            }
        });
        return prediction;
    }

    window.ALGORITHMS.linearRegression = linearRegressionConfig;
})();
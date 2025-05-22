// js/algorithms/svm.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};

    const svmConfig = {
        name: 'Support Vector Machine (SVM)',
        type: 'classifier',
        description: `SVMs are powerful classifiers that find an optimal hyperplane to separate classes.
                      'Linear' kernel attempts a direct linear separation. 
                      'RBF' (Radial Basis Function) kernel can map data to a higher dimension to find non-linear boundaries.
                      Training involves optimizing a margin (hinge loss for linear). This implementation uses a simplified 
                      sub-gradient descent for Linear SVM. Full RBF training is complex; this demo might simplify its application.`,
        params: [
            { 
                id: 'svmKernel', 
                label: 'Kernel Type', 
                type: 'select', 
                options: [
                    {value: 'linear', text: 'Linear'}, 
                    {value: 'rbf', text: 'RBF (Radial Basis Function)'}
                ],
                default: 'linear' 
            },
            { 
                id: 'svmC', 
                label: 'Regularization (C)', 
                type: 'number', 
                min: 0.01, 
                max: 10.0, 
                step: 0.01, 
                default: 1.0,
                condition: (params) => params.svmKernel === 'linear' // Only for linear for now
            },
            { 
                id: 'svmIterations', 
                label: 'Training Iterations (Linear)', 
                type: 'number', 
                min: 100, 
                max: 2000, 
                step: 50, 
                default: 500,
                condition: (params) => params.svmKernel === 'linear'
            },
            { 
                id: 'svmLearningRate', 
                label: 'Learning Rate (Linear)', 
                type: 'number', 
                min: 0.0001, 
                max: 0.1, 
                step: 0.0001, 
                default: 0.001,
                condition: (params) => params.svmKernel === 'linear'
            },
            { 
                id: 'svmGamma', 
                label: 'Gamma (RBF Kernel)', 
                type: 'number', 
                min: 0.01, 
                max: 10.0, 
                step: 0.01, 
                default: 1.0,
                condition: (params) => params.svmKernel === 'rbf'
            }
        ],
        train: trainSVM,
        predict: predictSVM,
        isIterative: true // For the linear SVM with gradient descent
    };

    function trainSVM(dataset, hyperparameters) {
        if (!dataset || dataset.length === 0) return null;

        const kernel = hyperparameters.svmKernel;
        
        // SVM targets are typically -1 and 1
        const mappedDataset = dataset.map(d => ({
            inputs: d.inputs,
            target: d.target === 0 ? -1 : 1 
        }));

        if (kernel === 'linear') {
            return trainLinearSVM(mappedDataset, hyperparameters);
        } else if (kernel === 'rbf') {
            // Full RBF SVM training is complex. 
            // For a demo, we might store the data and gamma, and prediction would use all points as potential SVs (like Kernel k-NN).
            // Or, if we had a QP solver or a library, we'd use it here.
            // This simplified version will just store data for kernelized prediction.
            console.warn("RBF SVM training in vanilla JS is highly simplified for this demo. It will behave like Kernel Regression / Kernel k-NN for prediction, not a true sparse SVM.");
            return { 
                data: mappedDataset, // Store all data points as potential support vectors
                gamma: hyperparameters.svmGamma, 
                kernel: 'rbf',
                type: 'classifier' 
                // True SVM would have: support_vectors, alphas, bias
            };
        }
        return null;
    }
    
    function trainLinearSVM(dataset, hyperparameters) {
        const numFeatures = dataset[0].inputs.length;
        let weights = Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.01);
        let bias = 0;

        const C = hyperparameters.svmC; // Regularization parameter
        const iterations = hyperparameters.svmIterations;
        const learningRate = hyperparameters.svmLearningRate; // Or eta, often decreases over time
        const N = dataset.length;

        for (let iter = 0; iter < iterations; iter++) {
            // Simple SGD update for Pegasos-like SVM
            // Pick a random sample (or iterate, but random is common for Pegasos)
            const sample = dataset[Math.floor(Math.random() * N)];
            
            let decisionValue = bias;
            sample.inputs.forEach((val, i) => decisionValue += weights[i] * val);

            // Hinge loss condition: y_i * (w . x_i + b) >= 1
            if (sample.target * decisionValue < 1) { 
                // Misclassified or within the margin, update
                bias += learningRate * C * sample.target; // Simplified: no 1/iter for eta
                weights.forEach((w, i) => {
                    weights[i] = (1 - learningRate/ (iter + 1)) * weights[i] + learningRate * C * sample.target * sample.inputs[i]; // Weight decay + update
                    // A simpler update without explicit weight decay in LR:
                    // weights[i] += learningRate * C * sample.target * sample.inputs[i];
                });
            } else {
                // Correctly classified and outside margin, only apply weight decay (Pegasos style)
                 weights.forEach((w, i) => {
                    weights[i] = (1 - learningRate/ (iter + 1)) * weights[i];
                });
            }
            // More common subgradient descent would iterate all points, sum gradients, then update.
            // This is a stochastic (Pegasos-like) variant.
        }
        // console.log("Linear SVM Trained:", {weights, bias});
        return { weights, bias, kernel: 'linear', type: 'classifier' };
    }


    function predictSVM(pointInputs, model) {
        if (!model) return -1;

        if (model.kernel === 'linear') {
            if (!model.weights || typeof model.bias === 'undefined') return -1;
            let decisionValue = model.bias;
            pointInputs.forEach((val, i) => {
                if (model.weights[i] !== undefined) decisionValue += model.weights[i] * val;
            });
            return decisionValue > 0 ? 1 : 0; // Map back to 0/1 for our app
        } else if (model.kernel === 'rbf') {
            // Simplified RBF prediction (like kernel k-NN or kernel regression, not true sparse SVM)
            // This requires access to all training data points, which are stored in model.data
            if (!model.data || !model.gamma) return -1;
            
            // This is NOT a true SVM prediction with alphas and support vectors.
            // It's a heuristic: sum of kernel influences from all training points.
            // A proper SVM prediction would be: sign(sum(alpha_i * y_i * K(sv_i, x)) + b)
            // We don't have alphas or a learned bias 'b' from this simplified RBF "training".

            // Let's implement a simple "kernelized vote" for RBF.
            // Calculate weighted sum of target values (-1, 1) based on RBF kernel similarity.
            let weightedSum = 0;
            let totalKernelSimilarity = 0; // For normalization if needed, or just use raw sum

            model.data.forEach(sv_like => { // sv_like are all training points here
                let sqDist = 0;
                pointInputs.forEach((val, i) => sqDist += (val - sv_like.inputs[i])**2);
                const kernelVal = Math.exp(-model.gamma * sqDist); // RBF kernel
                
                weightedSum += sv_like.target * kernelVal; // sv_like.target is -1 or 1
                totalKernelSimilarity += kernelVal;
            });
            
            // If totalKernelSimilarity is very small, it means point is far from all "support vectors"
            // if (totalKernelSimilarity < 1e-6 && model.data.length > 0) {
            //     // Fallback: predict based on the class of the single closest point in original space (like 1-NN)
            //     // This is a heuristic to handle points far from the data cloud in RBF.
            //     let minDist = Infinity;
            //     let closestTarget = model.data[0].target; // Default
            //     model.data.forEach(dp => {
            //         let d = 0; pointInputs.forEach((val, i) => d += (val - dp.inputs[i])**2);
            //         if (d < minDist) { minDist = d; closestTarget = dp.target; }
            //     });
            //     return closestTarget > 0 ? 1 : 0;
            // }

            return weightedSum > 0 ? 1 : 0; // Map back to 0/1 for our app
        }
        return -1;
    }

    window.ALGORITHMS.svm = svmConfig;
})();
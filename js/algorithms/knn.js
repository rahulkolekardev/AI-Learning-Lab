// js/algorithms/knn.js

(function() {
    // Ensure the global ALGORITHMS object exists
    window.ALGORITHMS = window.ALGORITHMS || {};

    const knnConfig = {
        name: 'k-Nearest Neighbors (k-NN)',
        type: 'classifier', // k-NN is primarily a classifier, can be adapted for regression
        description: `k-NN is a non-parametric, instance-based learning algorithm. 
                      It classifies a new data point based on the majority class of its 'k' closest neighbors 
                      in the feature space. The "training" phase simply involves storing the dataset. 
                      Key considerations include choosing an appropriate 'k' and a distance metric (Euclidean is used here). 
                      It can be computationally intensive for predictions on large datasets.`,
        params: [
            { 
                id: 'knnK', 
                label: 'Number of Neighbors (k)', 
                type: 'range', 
                min: 1, 
                // Max k should ideally not exceed number of samples, but we'll cap it for UI sanity
                // This will be dynamically adjusted in main.js if numSamples is too low
                max: 25, 
                step: 1, 
                default: 5 
            },
            // Future potential param: Distance Metric (Euclidean, Manhattan, etc.)
            // {
            //     id: 'knnDistanceMetric',
            //     label: 'Distance Metric',
            //     type: 'select',
            //     options: [{value: 'euclidean', text: 'Euclidean'}, {value: 'manhattan', text: 'Manhattan (Coming Soon)'}],
            //     default: 'euclidean'
            // }
        ],
        train: trainKNN,
        predict: predictKNN,
        isIterative: false // No iterative training process
    };

    function trainKNN(dataset, hyperparameters) {
        // For k-NN, "training" is just storing the dataset and parameters.
        // No model parameters like weights are learned.
        if (!dataset || dataset.length === 0) {
            console.warn("k-NN Training: Dataset is empty.");
            return null; // Or handle error appropriately
        }
        return { 
            data: dataset, // Store a reference to the training data
            k: parseInt(hyperparameters.knnK), // Ensure k is an integer
            // distanceMetric: hyperparameters.knnDistanceMetric || 'euclidean', // If distance metric option is added
            type: 'classifier' 
        };
    }

    function predictKNN(pointInputs, model) {
        if (!model || !model.data || model.data.length === 0) {
            // console.warn("k-NN Prediction: Model not trained or training data is empty.");
            return -1; // Or a default class / error indicator
        }
        if (typeof model.k !== 'number' || model.k < 1) {
            // console.warn(`k-NN Prediction: Invalid k value: ${model.k}`);
            return -1;
        }

        const k = Math.min(model.k, model.data.length); // k cannot be more than available data points

        // Calculate distances to all points in the training set
        const distances = model.data.map(dataPoint => {
            // Using Euclidean distance
            let sumOfSquares = 0;
            for (let i = 0; i < pointInputs.length; i++) {
                // Ensure dataPoint.inputs[i] exists if pointInputs has more dimensions
                const diff = (dataPoint.inputs[i] || 0) - (pointInputs[i] || 0);
                sumOfSquares += diff * diff;
            }
            return { 
                dist: Math.sqrt(sumOfSquares), 
                target: dataPoint.target 
            };
        });

        // Sort distances in ascending order
        distances.sort((a, b) => a.dist - b.dist);

        // Get the k nearest neighbors
        const neighbors = distances.slice(0, k);
        
        if (neighbors.length === 0) {
            return -1; // Should not happen if k >= 1 and data exists
        }

        // Perform a majority vote for classification
        const votes = {};
        neighbors.forEach(neighbor => {
            votes[neighbor.target] = (votes[neighbor.target] || 0) + 1;
        });

        let maxVotes = 0;
        let predictedClass = -1; // Default prediction if no clear majority or no neighbors
        let tie = false;
        let potentialTies = [];

        for (const classLabel in votes) {
            if (votes[classLabel] > maxVotes) {
                maxVotes = votes[classLabel];
                predictedClass = parseInt(classLabel);
                tie = false;
                potentialTies = [predictedClass];
            } else if (votes[classLabel] === maxVotes) {
                tie = true;
                potentialTies.push(parseInt(classLabel));
            }
        }
        
        // Handle ties: for simplicity, pick the first one in the tie, or the one with the closest neighbor.
        // A more robust tie-breaking might involve reducing k or using weighted distances.
        if (tie && potentialTies.length > 0) {
            // console.log("k-NN Tie encountered. Neighbors:", neighbors, "Votes:", votes, "Potential Ties:", potentialTies);
            // Simplest tie-break: pick the class of the absolute nearest neighbor among the tied classes.
            for (const neighbor of neighbors) {
                if (potentialTies.includes(neighbor.target)) {
                    predictedClass = neighbor.target;
                    break;
                }
            }
        }
        
        return predictedClass;
    }

    // Expose the configuration to the global scope
    window.ALGORITHMS.knn = knnConfig;

})();
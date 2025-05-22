// js/algorithms/naiveBayes.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};

    const naiveBayesConfig = {
        name: 'Naive Bayes (Gaussian)',
        type: 'classifier',
        description: `Naive Bayes is a probabilistic classifier based on Bayes' theorem with a 'naive' assumption 
                      of conditional independence between features. This version uses Gaussian Naive Bayes, 
                      suitable for continuous numerical features, assuming they follow a Gaussian (normal) distribution 
                      for each class. It's fast to train and often performs well, especially with high-dimensional data.`,
        params: [
            // No hyperparameters typically tuned by users for basic Gaussian Naive Bayes in demos.
            // Laplace/Lidstone smoothing (additive smoothing) could be one for categorical, but not for Gaussian here.
        ],
        train: trainGaussianNaiveBayes,
        predict: predictGaussianNaiveBayes,
        isIterative: false
    };

    // Helper function to calculate mean
    function mean(numbers) {
        return numbers.reduce((sum, val) => sum + val, 0) / numbers.length;
    }

    // Helper function to calculate variance (sample variance, N-1 denominator for unbiased estimate)
    function variance(numbers, precomputedMean) {
        const m = precomputedMean !== undefined ? precomputedMean : mean(numbers);
        if (numbers.length < 2) return 0; // Or handle as error/NaN, variance needs at least 2 points
        return numbers.reduce((sum, val) => sum + (val - m) ** 2, 0) / (numbers.length - 1);
    }
    
    // Helper function to calculate standard deviation
    function stdev(numbers, precomputedMean, precomputedVariance) {
        if (precomputedVariance !== undefined) {
            return Math.sqrt(precomputedVariance);
        }
        return Math.sqrt(variance(numbers, precomputedMean));
    }


    // Gaussian Probability Density Function
    function calculateGaussianProbability(x, meanVal, stdevVal) {
        if (stdevVal === 0) { // Avoid division by zero; if stdev is 0, it's a spike or all values are same
            return (x === meanVal) ? 1 : 1e-9; // High prob if x is exactly the mean, else very low
        }
        const exponent = Math.exp(-((x - meanVal) ** 2) / (2 * stdevVal ** 2));
        return (1 / (Math.sqrt(2 * Math.PI) * stdevVal)) * exponent;
    }

    function trainGaussianNaiveBayes(dataset, hyperparameters) {
        if (!dataset || dataset.length === 0) return null;

        const summariesByClass = {}; // { classVal: [{mean, stdev, count, prior}, ...for each feature], ... }
        const classPriors = {};
        const numFeatures = dataset[0].inputs.length;

        // Separate data by class
        const separatedByClass = {};
        dataset.forEach(instance => {
            if (!separatedByClass[instance.target]) {
                separatedByClass[instance.target] = [];
            }
            separatedByClass[instance.target].push(instance.inputs);
        });

        for (const classValue in separatedByClass) {
            const classData = separatedByClass[classValue]; // Array of input arrays for this class
            summariesByClass[classValue] = [];
            classPriors[classValue] = classData.length / dataset.length;

            for (let i = 0; i < numFeatures; i++) {
                const featureValuesForClass = classData.map(inputs => inputs[i]);
                const featureMean = mean(featureValuesForClass);
                let featureStdev = stdev(featureValuesForClass, featureMean);
                
                // Add a small epsilon to stdev if it's zero to prevent division by zero in PDF
                // and to handle cases where all feature values for a class are identical.
                if (featureStdev < 1e-9) { // A very small standard deviation is practically zero
                    featureStdev = 1e-9; 
                }

                summariesByClass[classValue].push({
                    mean: featureMean,
                    stdev: featureStdev,
                });
            }
        }
        
        // console.log("Naive Bayes Summaries:", summariesByClass, "Priors:", classPriors);
        return { summaries: summariesByClass, priors: classPriors, type: 'classifier' };
    }

    function predictGaussianNaiveBayes(pointInputs, model) {
        if (!model || !model.summaries || !model.priors) return -1;

        const probabilities = {};
        for (const classValue in model.summaries) {
            probabilities[classValue] = Math.log(model.priors[classValue]); // Start with log of prior

            model.summaries[classValue].forEach((featureSummary, featureIndex) => {
                const x = pointInputs[featureIndex];
                // Add log of likelihood P(feature_i | class)
                // Using logs helps prevent underflow with many small probabilities.
                const likelihood = calculateGaussianProbability(x, featureSummary.mean, featureSummary.stdev);
                probabilities[classValue] += Math.log(likelihood + 1e-9); // Add epsilon to avoid log(0)
            });
        }

        let bestLabel = -1;
        let bestProb = -Infinity;
        for (const classValue in probabilities) {
            if (probabilities[classValue] > bestProb) {
                bestProb = probabilities[classValue];
                bestLabel = parseInt(classValue);
            }
        }
        return bestLabel;
    }

    window.ALGORITHMS.naiveBayes = naiveBayesConfig;
})();
// js/algorithms/randomForest.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};
    // Depends on Decision Tree algorithm being loaded if we reuse its functions directly.
    // For true isolation, decision tree logic would be duplicated or passed as dependency.
    // Assuming ALGORITHMS.decisionTree.train and ALGORITHMS.decisionTree.predict are available.

    const randomForestConfig = {
        name: 'Random Forest (Classifier)',
        type: 'classifier',
        description: `Random Forest is an ensemble learning method that operates by constructing multiple 
                      Decision Trees at training time and outputting the class that is the mode of the classes 
                      (classification) of the individual trees. It typically improves accuracy and reduces overfitting 
                      compared to a single decision tree. This is a simplified version.`,
        params: [
            { 
                id: 'rfNumTrees', 
                label: 'Number of Trees', 
                type: 'range', 
                min: 5, 
                max: 50, // Keep it reasonable for browser performance
                step: 1, 
                default: 10 
            },
            { 
                id: 'rfMaxDepth', 
                label: 'Max Depth (per Tree)', 
                type: 'range', 
                min: 1, 
                max: 10, 
                step: 1, 
                default: 4 
            },
            { 
                id: 'rfMinSamplesSplit', 
                label: 'Min Samples Split (per Tree)', 
                type: 'range', 
                min: 2, 
                max: 20, 
                step: 1, 
                default: 2 
            },
            {
                id: 'rfSampleSizeRatio',
                label: 'Bootstrap Sample Ratio (per Tree)',
                type: 'number',
                min: 0.1, max: 1.0, step: 0.1, default: 0.7
            }
            // Future: Max Features per split
        ],
        train: trainRandomForest,
        predict: predictRandomForest,
        isIterative: false // Each tree is built, then ensemble is formed
    };

    function bootstrapSample(dataset, ratio) {
        const sample = [];
        const n_sample = Math.round(dataset.length * ratio);
        for (let i = 0; i < n_sample; i++) {
            const index = Math.floor(Math.random() * dataset.length);
            sample.push(dataset[index]);
        }
        return sample;
    }

    function trainRandomForest(dataset, hyperparameters) {
        if (!dataset || dataset.length === 0) return null;
        if (!window.ALGORITHMS.decisionTree || !window.ALGORITHMS.decisionTree.train) {
            console.error("Random Forest requires Decision Tree algorithm to be loaded.");
            return null;
        }

        const trees = [];
        const numTrees = hyperparameters.rfNumTrees;
        const sampleRatio = hyperparameters.rfSampleSizeRatio;

        // Decision Tree hyperparameters for each tree in the forest
        const treeHyperparams = {
            dtMaxDepth: hyperparameters.rfMaxDepth,
            dtMinSamplesSplit: hyperparameters.rfMinSamplesSplit
        };

        for (let i = 0; i < numTrees; i++) {
            const sample = bootstrapSample(dataset, sampleRatio);
            if (sample.length < treeHyperparams.dtMinSamplesSplit) { // Ensure sample is large enough
                // console.warn(`Bootstrap sample too small for tree ${i}, skipping.`);
                continue;
            }
            // For a true Random Forest, feature sub-sampling at each split node would be done
            // within the decision tree's getBestSplit function.
            // Our current Decision Tree doesn't support that, so we simplify.
            const treeModel = window.ALGORITHMS.decisionTree.train(sample, treeHyperparams);
            if (treeModel && treeModel.tree) { // Ensure a valid tree was built
                 trees.push(treeModel.tree);
            }
        }
        // console.log(`Random Forest trained with ${trees.length} trees.`);
        return { trees, type: 'classifier' };
    }

    function predictRandomForest(pointInputs, model) {
        if (!model || !model.trees || model.trees.length === 0) return -1;
        if (!window.ALGORITHMS.decisionTree || !window.ALGORITHMS.decisionTree.predict) {
            console.error("Random Forest prediction requires Decision Tree algorithm.");
            return -1;
        }

        const predictions = model.trees.map(tree => {
            // The predict function for Decision Tree needs the full model object {tree: treeNode}
            return window.ALGORITHMS.decisionTree.predict(pointInputs, { tree: tree });
        });

        // Majority vote
        const votes = {};
        predictions.forEach(pred => {
            if (pred !== undefined && pred !== null && pred !== -1) { // Ignore invalid predictions
                 votes[pred] = (votes[pred] || 0) + 1;
            }
        });

        let maxVotes = 0;
        let majorityClass = -1; // Default
         if (Object.keys(votes).length === 0 && predictions.length > 0) {
            // If all tree predictions were invalid, maybe return a default or error
            // For now, if predictions array had items but votes is empty, means all predictions were invalid
            // console.warn("Random Forest: All individual tree predictions were invalid.");
            return predictions.length > 0 ? predictions[0] : -1; // Fallback to first tree's (potentially invalid) prediction
        }

        for (const classLabel in votes) {
            if (votes[classLabel] > maxVotes) {
                maxVotes = votes[classLabel];
                majorityClass = parseInt(classLabel);
            }
        }
        return majorityClass;
    }

    window.ALGORITHMS.randomForest = randomForestConfig;
})();
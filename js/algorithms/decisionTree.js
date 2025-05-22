// js/algorithms/decisionTree.js

(function() {
    window.ALGORITHMS = window.ALGORITHMS || {};

    const decisionTreeConfig = {
        name: 'Decision Tree (Classifier)',
        type: 'classifier',
        description: `A Decision Tree is a non-parametric supervised learning method that predicts the value of a target 
                      variable by learning simple decision rules inferred from the data features. It partitions the data space 
                      into regions, with each leaf node representing a class label. This implementation uses Gini Impurity 
                      to measure the quality of a split. Tree growth is controlled by 'Max Depth' and 
                      'Min Samples for Split' to prevent overfitting.`,
        params: [
            { 
                id: 'dtMaxDepth', 
                label: 'Maximum Tree Depth', 
                type: 'range', 
                min: 1, 
                max: 15, // Max depth can significantly impact complexity
                step: 1, 
                default: 5 
            },
            { 
                id: 'dtMinSamplesSplit', 
                label: 'Min Samples for Split', 
                type: 'range', 
                min: 2, 
                max: 30, // Adjust based on typical dataset sizes
                step: 1, 
                default: 2 
            }
            // Future: Criterion (Gini/Entropy), Min Samples Leaf
        ],
        train: trainDecisionTree,
        predict: predictDecisionTreeRecursive, // Using the recursive prediction helper
        isIterative: false // Tree building is typically a one-pass (recursive) process
    };

    // --- Helper Functions for Decision Tree ---

    /**
     * Calculates the Gini impurity for a list of groups of samples.
     * Gini impurity is a measure of how often a randomly chosen element
     * from the set would be incorrectly labeled if it was randomly labeled
     * according to the distribution of labels in the subset.
     * Gini = 1 - sum(p_i^2) for each class i.
     * Weighted average Gini for groups.
     */
    function calculateGiniImpurity(groups, classes) {
        let totalInstances = 0;
        groups.forEach(group => totalInstances += group.length);

        if (totalInstances === 0) return 1; // Max impurity if no instances

        let gini = 0.0;
        groups.forEach(group => {
            if (group.length === 0) return;

            let score = 0.0;
            classes.forEach(classVal => {
                const proportion = group.filter(row => row.target === classVal).length / group.length;
                score += proportion * proportion;
            });
            gini += (1.0 - score) * (group.length / totalInstances);
        });
        return gini;
    }

    /**
     * Splits a dataset based on an attribute and an attribute value.
     */
    function testSplit(featureIndex, value, dataset) {
        const left = [], right = [];
        dataset.forEach(row => {
            if (row.inputs[featureIndex] < value) {
                left.push(row);
            } else {
                right.push(row);
            }
        });
        return { left, right };
    }

    /**
     * Selects the best split point for a dataset.
     * Iterates over all features and all unique values for each feature,
     * calculating the Gini impurity for each potential split.
     */
    function getBestSplit(dataset) {
        if (!dataset || dataset.length === 0 || !dataset[0].inputs) {
            return { index: -1, value: undefined, groups: null, score: Infinity };
        }
        const numFeatures = dataset[0].inputs.length;
        const classValues = [...new Set(dataset.map(row => row.target))];
        
        let bestSplit = { index: -1, value: undefined, groups: null, score: Infinity };

        for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            // Create a sorted list of unique values for the current feature to test as split points
            const uniqueFeatureValues = [...new Set(dataset.map(row => row.inputs[featureIndex]))].sort((a, b) => a - b);
            
            for (let i = 0; i < uniqueFeatureValues.length -1; i++) {
                // Use midpoint between unique sorted values as potential split threshold
                const splitValue = (uniqueFeatureValues[i] + uniqueFeatureValues[i+1]) / 2.0;
                const groups = testSplit(featureIndex, splitValue, dataset);
                const gini = calculateGiniImpurity([groups.left, groups.right], classValues);

                if (gini < bestSplit.score) {
                    bestSplit.index = featureIndex;
                    bestSplit.value = splitValue;
                    bestSplit.groups = groups;
                    bestSplit.score = gini;
                }
            }
             // Also consider splitting at the unique values themselves (or just below/above)
            // This is more robust if midpoints are not ideal (e.g. very sparse data)
            // For simplicity, we'll stick to midpoints for now
        }
        return bestSplit;
    }

    /**
     * Creates a terminal node value (majority class in the group).
     */
    function toTerminalNode(group) {
        if (group.length === 0) return undefined; // Should not happen if minSamplesSplit >= 1

        const outcomes = group.map(row => row.target);
        const counts = {};
        let maxCount = 0;
        let majorityClass = outcomes[0]; // Default to first if all unique or empty

        outcomes.forEach(outcome => {
            counts[outcome] = (counts[outcome] || 0) + 1;
            if (counts[outcome] > maxCount) {
                maxCount = counts[outcome];
                majorityClass = outcome;
            }
        });
        return majorityClass;
    }

    /**
     * Recursive helper to build the decision tree.
     */
    function buildTreeRecursive(node, maxDepth, minSamplesSplit, currentDepth) {
        const { left, right } = node.groups; // Groups from best split
        delete node.groups; // No longer need to store full datasets in the node

        // Check for no split (all data went to one side, or pure node)
        if (!left || !right || left.length === 0 || right.length === 0) {
            // Make this node a terminal node with combined group
            node.leftLeaf = node.rightLeaf = toTerminalNode(left.concat(right));
            return;
        }

        // Check for max depth
        if (currentDepth >= maxDepth) {
            node.leftLeaf = toTerminalNode(left);
            node.rightLeaf = toTerminalNode(right);
            return;
        }

        // Process left child
        if (left.length <= minSamplesSplit) {
            node.leftLeaf = toTerminalNode(left);
        } else {
            node.leftNode = getBestSplit(left); // Find best split for left child
            if (node.leftNode.score === Infinity) { // No further beneficial split
                 node.leftLeaf = toTerminalNode(left);
            } else {
                 buildTreeRecursive(node.leftNode, maxDepth, minSamplesSplit, currentDepth + 1);
            }
        }

        // Process right child
        if (right.length <= minSamplesSplit) {
            node.rightLeaf = toTerminalNode(right);
        } else {
            node.rightNode = getBestSplit(right); // Find best split for right child
             if (node.rightNode.score === Infinity) { // No further beneficial split
                 node.rightLeaf = toTerminalNode(right);
            } else {
                buildTreeRecursive(node.rightNode, maxDepth, minSamplesSplit, currentDepth + 1);
            }
        }
    }

    // --- Main Training and Prediction Functions ---
    function trainDecisionTree(dataset, hyperparameters) {
        if (!dataset || dataset.length === 0) {
            console.warn("Decision Tree Training: Dataset is empty.");
            return null;
        }
        const maxDepth = hyperparameters.dtMaxDepth;
        const minSamplesSplit = hyperparameters.dtMinSamplesSplit;

        const root = getBestSplit(dataset);
        
        if (root.score === Infinity || !root.groups) { // No split possible or dataset too small/pure
            // console.log("Decision Tree: Root node is terminal or no split possible.");
            return { tree: { value: toTerminalNode(dataset) }, type: 'classifier' }; // Tree is just a single leaf
        }

        buildTreeRecursive(root, maxDepth, minSamplesSplit, 1);
        // console.log("Trained Decision Tree:", JSON.stringify(root, null, 2));
        return { tree: root, type: 'classifier' };
    }

    function predictDecisionTreeRecursive(inputs, node) {
        // If node is a direct value (leaf prediction)
        if (typeof node !== 'object' || node === null || (typeof node.leftLeaf === 'undefined' && typeof node.rightLeaf === 'undefined' && typeof node.leftNode === 'undefined' && typeof node.rightNode === 'undefined')) {
            // This means 'node' itself is the predicted class (e.g. from toTerminalNode)
            // Or it's a malformed leaf from a no-split scenario in root.
            return (typeof node === 'object' && node !== null && typeof node.value !== 'undefined') ? node.value : node;
        }

        // Check if current node is a leaf due to depth/min_samples or purity
        if (inputs[node.index] < node.value) { // Go left
            if (typeof node.leftLeaf !== 'undefined') {
                return node.leftLeaf;
            } else if (node.leftNode) {
                return predictDecisionTreeRecursive(inputs, node.leftNode);
            } else { // Should have either leftLeaf or leftNode if it's a split node
                // Fallback if tree structure is imperfect (e.g. one branch didn't split further)
                // This might happen if a split leads to an empty group that wasn't properly terminalized.
                // console.warn("DT Predict: Inconsistent left branch at node:", node);
                return undefined; // Or a default prediction
            }
        } else { // Go right
            if (typeof node.rightLeaf !== 'undefined') {
                return node.rightLeaf;
            } else if (node.rightNode) {
                return predictDecisionTreeRecursive(inputs, node.rightNode);
            } else {
                // console.warn("DT Predict: Inconsistent right branch at node:", node);
                return undefined;
            }
        }
    }

    // Expose the configuration to the global scope
    window.ALGORITHMS.decisionTree = decisionTreeConfig;

})();
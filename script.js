    // --- Polyfills for Math.log2 if needed (older browsers) ---
    Math.log2 = Math.log2 || function(x) { return Math.log(x) * Math.LOG2E; };

    document.addEventListener('DOMContentLoaded', () => {
        // --- DOM Elements ---
        const algorithmSelect = document.getElementById('algorithmSelect');
        const datasetTypeSelect = document.getElementById('datasetType');
        const numSamplesSlider = document.getElementById('numSamples');
        const numSamplesValueSpan = document.getElementById('numSamplesValue');
        const noiseLevelSlider = document.getElementById('noiseLevel');
        const noiseLevelValueSpan = document.getElementById('noiseLevelValue');
        const generateDataButton = document.getElementById('generateDataButton');
        const hyperparamsContainer = document.getElementById('hyperparamsContainer');
        const trainButton = document.getElementById('trainButton');
        const resetButton = document.getElementById('resetButton');

        const decisionBoundaryCanvas = document.getElementById('decisionBoundaryCanvas');
        const dbCtx = decisionBoundaryCanvas.getContext('2d');
        const dataPointsCanvas = document.getElementById('dataPointsCanvas');
        const dpCtx = dataPointsCanvas.getContext('2d');
        const visualizationCaption = document.getElementById('visualizationCaption');

        const metricAlgorithmSpan = document.getElementById('metricAlgorithm');
        const metricAccuracySpan = document.getElementById('metricAccuracy');
        const accuracyMetricItem = document.getElementById('accuracyMetricItem');
        const metricMSESpan = document.getElementById('metricMSE');
        const mseMetricItem = document.getElementById('mseMetricItem');

        const infoAlgorithmNameSpan = document.getElementById('infoAlgorithmName');
        const infoAlgorithmDescriptionP = document.getElementById('infoAlgorithmDescription');

        // --- App State ---
        let currentDataset = []; 
        let currentAlgorithm = algorithmSelect.value;
        let trainedModel = null; 
        let hyperparameters = {};

        const ALGORITHM_CONFIG = {
            'knn': {
                name: 'k-Nearest Neighbors', type: 'classifier',
                description: 'k-NN is a non-parametric algorithm that classifies data points based on the majority class of their k-closest neighbors in the feature space. Training involves storing the dataset; prediction is computationally intensive for large datasets.',
                params: [ { id: 'knnK', label: 'Neighbor Count (k)', type: 'range', min: 1, max: 25, step: 1, default: 5 } ],
                train: trainKNN, predict: predictKNN, isIterative: false
            },
            'linearRegression': {
                name: 'Linear Regression', type: 'regressor',
                description: 'Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. This implementation uses Gradient Descent for optimization.',
                params: [
                    { id: 'lrLearningRate', label: 'Learning Rate (α)', type: 'number', min:0.0001, max:0.1, step:0.0001, default: 0.01 },
                    { id: 'lrIterations', label: 'Training Iterations', type: 'number', min:50, max:1000, step:10, default: 200 }
                ],
                train: trainLinearRegression, predict: predictLinearRegression, isIterative: true
            },
            'logisticRegression': {
                name: 'Logistic Regression', type: 'classifier',
                description: 'Logistic Regression is a statistical model for binary classification. It uses a logistic (sigmoid) function to model the probability of a dichotomous outcome. Optimized via Gradient Descent.',
                params: [
                    { id: 'logRegLearningRate', label: 'Learning Rate (α)', type: 'number', min:0.001, max:1, step:0.001, default: 0.1 },
                    { id: 'logRegIterations', label: 'Training Iterations', type: 'number', min:100, max:2000, step:50, default: 500 }
                ],
                train: trainLogisticRegression, predict: predictLogisticRegression, isIterative: true
            },
             'decisionTree': {
                name: 'Decision Tree', type: 'classifier',
                description: 'Decision Trees learn a hierarchy of if/else questions, leading to a decision. This implementation uses Gini impurity to find optimal splits and is pruned by max depth and minimum samples per split.',
                params: [
                    { id: 'dtMaxDepth', label: 'Maximum Depth', type: 'range', min: 1, max: 10, step: 1, default: 4 },
                    { id: 'dtMinSamplesSplit', label: 'Min Samples for Split', type: 'range', min: 2, max: 20, step: 1, default: 2 }
                ],
                train: trainDecisionTree, predict: predictDecisionTree, isIterative: false
            }
        };

        // --- Dataset Generation --- (Identical JS logic, only UI colors change with CSS)
        function generateDataset() {
            const type = datasetTypeSelect.value;
            const n = parseInt(numSamplesSlider.value);
            const noise = parseFloat(noiseLevelSlider.value);
            currentDataset = [];
            const isRegression = ALGORITHM_CONFIG[currentAlgorithm]?.type === 'regressor';

            function randn_bm() { let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random(); return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v); }
            const rNoise = (scale = 1) => (Math.random() - 0.5) * 2 * noise * scale;

            for (let i = 0; i < n; i++) {
                let x_feat, y_feat, target_val_or_class; // Use more generic names
                x_feat = Math.random(); // Primary feature, 0 to 1

                switch (type) {
                    case 'circles':
                        const r_circle = Math.random() > 0.5 ? 0.22 : 0.42; // Adjusted radii for better visual separation
                        const angle_circle = Math.random() * 2 * Math.PI;
                        x_feat = 0.5 + r_circle * Math.cos(angle_circle) + rNoise(0.4);
                        y_feat = 0.5 + r_circle * Math.sin(angle_circle) + rNoise(0.4);
                        target_val_or_class = r_circle > 0.3 ? 1 : 0;
                        currentDataset.push({ inputs: [norm(x_feat), norm(y_feat)], target: target_val_or_class });
                        break;
                    case 'xor':
                        x_feat = Math.random() > 0.5 ? 0.75 : 0.25;
                        y_feat = Math.random() > 0.5 ? 0.75 : 0.25;
                        target_val_or_class = (x_feat > 0.5) ^ (y_feat > 0.5) ? 1 : 0;
                        currentDataset.push({ inputs: [norm(x_feat + rNoise(0.25)), norm(y_feat + rNoise(0.25))], target: target_val_or_class });
                        break;
                    case 'linearTrend':
                        y_feat = 0.2 + 0.6 * x_feat + rNoise(0.25); // y_feat is the target here
                        currentDataset.push({ inputs: [x_feat], target: y_feat });
                        break;
                    case 'sineWave':
                        y_feat = 0.5 + 0.3 * Math.sin(x_feat * 2.5 * Math.PI) + rNoise(0.15); // y_feat is target
                        currentDataset.push({ inputs: [x_feat], target: y_feat });
                        break;
                    case 'gauss':
                    default:
                        const mean1 = [0.3, 0.3], mean2 = [0.7, 0.7]; const std_gauss = 0.1;
                        if (i < n / 2) {
                            x_feat = mean1[0] + randn_bm() * std_gauss + rNoise(0.2);
                            y_feat = mean1[1] + randn_bm() * std_gauss + rNoise(0.2);
                            target_val_or_class = 0;
                        } else {
                            x_feat = mean2[0] + randn_bm() * std_gauss + rNoise(0.2);
                            y_feat = mean2[1] + randn_bm() * std_gauss + rNoise(0.2);
                            target_val_or_class = 1;
                        }
                        currentDataset.push({ inputs: [norm(x_feat), norm(y_feat)], target: target_val_or_class });
                        break;
                }
            }
            function norm(val){ return Math.max(0.01, Math.min(0.99, val)); } // Avoid exact 0/1 for some algos

            trainedModel = null;
            drawDataPoints();
            clearDecisionBoundary();
            updateMetricsDisplay(true);
            updateVisualizationCaption();
        }
        
        // --- Algorithm Implementations (Identical JS logic) ---
        // k-NN (Classifier)
        function trainKNN(dataset, params) { return { data: dataset, k: params.knnK }; }
        function predictKNN(pointInputs, model) {
            const distances = model.data.map(dp => ({
                dist: Math.sqrt(dp.inputs.reduce((sum, val, i) => sum + (val - pointInputs[i])**2, 0)),
                target: dp.target
            }));
            distances.sort((a, b) => a.dist - b.dist);
            const neighbors = distances.slice(0, model.k);
            const votes = {}; neighbors.forEach(n => votes[n.target] = (votes[n.target]||0)+1);
            let maxV = 0, predC = -1; // Default if no majority (e.g. k=0 or empty neighbors)
            let hasVotes = false;
            for(const c_str in votes) {
                hasVotes = true;
                const c_int = parseInt(c_str);
                if(votes[c_str] > maxV){ maxV=votes[c_str]; predC=c_int; }
                // Tie-breaking: prefer smaller class index or random (not implemented here for simplicity)
            }
            return hasVotes ? predC : (model.data.length > 0 ? model.data[0].target : 0) ; // Fallback if no votes
        }

        // Linear Regression
        function trainLinearRegression(dataset, params) {
            let weights = Array(dataset[0].inputs.length).fill(0).map(() => Math.random()*0.02-0.01);
            let bias = 0;
            const lr = params.lrLearningRate; const iterations = params.lrIterations;
            for (let i = 0; i < iterations; i++) {
                let gradW = Array(weights.length).fill(0); let gradB = 0;
                dataset.forEach(s => {
                    const pred = s.inputs.reduce((sum,inp,idx)=>sum+inp*weights[idx],bias);
                    const err = pred - s.target;
                    gradB += err; s.inputs.forEach((inp,idx)=>gradW[idx]+=err*inp);
                });
                bias -= lr*(gradB/dataset.length);
                weights.forEach((w,idx)=>weights[idx]-=lr*(gradW[idx]/dataset.length));
            }
            return { weights, bias, type: 'regressor' };
        }
        function predictLinearRegression(pointInputs, model) {
            return pointInputs.reduce((sum,inp,idx)=>sum+inp*model.weights[idx],model.bias);
        }
        
        // Logistic Regression
        function sigmoid(z) { return 1/(1+Math.exp(-z)); }
        function trainLogisticRegression(dataset, params) {
            let weights = Array(dataset[0].inputs.length).fill(0).map(()=>Math.random()*0.02-0.01);
            let bias = 0;
            const lr = params.logRegLearningRate; const iterations = params.logRegIterations;
            for(let i=0; i<iterations; i++){
                let gradW=Array(weights.length).fill(0); let gradB=0;
                dataset.forEach(s=>{
                    const z = s.inputs.reduce((sum,inp,idx)=>sum+inp*weights[idx],bias);
                    const pred = sigmoid(z); const err = pred - s.target;
                    gradB += err; s.inputs.forEach((inp,idx)=>gradW[idx]+=err*inp);
                });
                bias -= lr*(gradB/dataset.length);
                weights.forEach((w,idx)=>weights[idx]-=lr*(gradW[idx]/dataset.length));
            }
            return { weights, bias, type: 'classifier' };
        }
        function predictLogisticRegression(pointInputs, model) {
            const z = pointInputs.reduce((sum,inp,idx)=>sum+inp*model.weights[idx],model.bias);
            return sigmoid(z) > 0.5 ? 1 : 0;
        }

        // Decision Tree
        function calculateGini(groups, classes) { /* ... (same) ... */ 
            let nInst = groups.reduce((s,g)=>s+g.length,0); if(nInst===0) return 1; let gini=0;
            groups.forEach(g=>{if(g.length===0)return; let score=0;
            classes.forEach(c=>{const p=g.filter(r=>r.target===c).length/g.length; score+=p*p;});
            gini+=(1.0-score)*(g.length/nInst);}); return gini;
        }
        function splitDataset(idx,val,ds){const l=[],r=[];ds.forEach(row=>{if(row.inputs[idx]<val)l.push(row);else r.push(row);});return{left:l,right:r};}
        function getBestSplit(ds) { /* ... (same, ensure classes are distinct) ... */
            const classes = [...new Set(ds.map(r=>r.target))]; let best={score:Infinity};
            if(ds.length===0||ds[0].inputs.length===0) return {...best, index:-1,value:-1,groups:null};
            for(let featIdx=0; featIdx<ds[0].inputs.length; featIdx++){
                const vals=[...new Set(ds.map(r=>r.inputs[featIdx]))].sort((a,b)=>a-b);
                for(let i=0;i<vals.length-1;i++){const val=(vals[i]+vals[i+1])/2;
                const grps=splitDataset(featIdx,val,ds); const gini=calculateGini([grps.left,grps.right],classes);
                if(gini<best.score){best={index:featIdx,value:val,score:gini,groups:grps};}}}
            return best;
        }
        function toTerminal(grp){if(grp.length===0)return undefined; const outs=grp.map(r=>r.target); return outs.sort((a,b)=>outs.filter(v=>v===a).length-outs.filter(v=>v===b).length).pop();}
        function recursiveSplit(node,maxD,minS,depth){
            const{left,right}=node.groups; delete node.groups;
            if(!left||!right||left.length===0||right.length===0){node.left=node.right=toTerminal(left.concat(right));return;}
            if(depth>=maxD){node.left=toTerminal(left);node.right=toTerminal(right);return;}
            if(left.length<=minS){node.left=toTerminal(left);}else{node.left=getBestSplit(left);recursiveSplit(node.left,maxD,minS,depth+1);}
            if(right.length<=minS){node.right=toTerminal(right);}else{node.right=getBestSplit(right);recursiveSplit(node.right,maxD,minS,depth+1);}
        }
        function trainDecisionTree(ds,params){const root=getBestSplit(ds); if(root.index !== -1) recursiveSplit(root,params.dtMaxDepth,params.dtMinSamplesSplit,1); else root.value = toTerminal(ds); /* Handle case of no split */ return{tree:root,type:'classifier'};}
        function predictDecisionTree(inputs,model){
            let node=model.tree;
            while(typeof node.left === 'object' && node.left !== null && // Check if left is a node object
                  typeof node.right === 'object' && node.right !== null){ // Check if right is a node object
                if(node.index===-1 || typeof node.index === 'undefined') return node.value; // Should be terminal
                if(inputs[node.index]<node.value) node=node.left; else node=node.right;
            }
            return (typeof node === 'object' && node !== null && typeof node.value !== 'undefined') ? node.value : node; // Handles cases where node itself is the prediction or a terminal node object
        }

        // --- UI & Control Logic (Identical JS logic) ---
        function updateHyperparameterUI() { hyperparamsContainer.innerHTML = ''; const config = ALGORITHM_CONFIG[currentAlgorithm]; if (!config || !config.params) return; config.params.forEach(param => { const group = document.createElement('div'); group.className = 'input-group'; const label = document.createElement('label'); label.setAttribute('for', param.id); const valueSpanId = `${param.id}Value`; const isRange = param.type === 'range'; label.innerHTML = `${param.label}: ${isRange ? `<span id="${valueSpanId}" class="range-value">${param.default}</span>` : ''}`; const input = document.createElement('input'); input.type = param.type === 'range' ? 'range' : 'number'; input.id = param.id; input.name = param.id; if (isRange || input.type === 'number') { input.min = param.min; input.max = param.max; input.step = param.step; } input.value = param.default; hyperparameters[param.id] = param.type === 'number' ? parseFloat(param.default) : param.default; input.addEventListener('input', (e) => { const val = param.type === 'number' ? parseFloat(e.target.value) : (isRange ? parseInt(e.target.value) : e.target.value); hyperparameters[param.id] = val; if (isRange) document.getElementById(valueSpanId).textContent = e.target.value; trainedModel = null; clearDecisionBoundary(); updateMetricsDisplay(true); }); group.appendChild(label); group.appendChild(input); hyperparamsContainer.appendChild(group); }); infoAlgorithmNameSpan.textContent = config.name; infoAlgorithmDescriptionP.textContent = config.description; metricAlgorithmSpan.textContent = config.name; mseMetricItem.style.display = config.type === 'regressor' ? 'flex' : 'none'; accuracyMetricItem.style.display = config.type === 'classifier' ? 'flex' : 'none'; updateVisualizationCaption(); }
        function handleAlgorithmChange() { currentAlgorithm = algorithmSelect.value; updateHyperparameterUI(); trainedModel = null; clearDecisionBoundary(); updateMetricsDisplay(true); const algoType = ALGORITHM_CONFIG[currentAlgorithm]?.type; const currentDataType = datasetTypeSelect.value.includes("Trend") || datasetTypeSelect.value.includes("Wave") ? 'regressor' : 'classifier'; if (algoType !== currentDataType) { if (algoType === 'regressor') { if (!datasetTypeSelect.value.includes("Trend") && !datasetTypeSelect.value.includes("Wave")) { datasetTypeSelect.value = 'linearTrend'; } } else { if (datasetTypeSelect.value.includes("Trend") || datasetTypeSelect.value.includes("Wave")) { datasetTypeSelect.value = 'gauss'; } } generateDataset(); } }
        function trainSelectedModel() { if (currentDataset.length === 0) { alert("Please generate a dataset first."); return; } const config = ALGORITHM_CONFIG[currentAlgorithm]; if (config && config.train) { trainButton.disabled = true; trainButton.textContent = "TRAINING..."; setTimeout(() => { try { trainedModel = config.train(currentDataset, hyperparameters); evaluateModel(); drawVisualization(); } catch (error) { console.error("Training error:", error); alert("An error occurred during training. Check console."); } finally { trainButton.disabled = false; trainButton.textContent = "RE-INITIATE TRAINING"; } }, 20); } else { alert("Selected algorithm training not implemented yet."); } }
        function evaluateModel() { if (!trainedModel || currentDataset.length === 0) { updateMetricsDisplay(true); return; } const config = ALGORITHM_CONFIG[currentAlgorithm]; if (!config || !config.predict) return; if (config.type === 'classifier') { let correct = 0; currentDataset.forEach(sample => { const prediction = config.predict(sample.inputs, trainedModel); if (prediction === sample.target) correct++; }); const accuracy = currentDataset.length > 0 ? (correct/currentDataset.length) : 0; updateMetricsDisplay(false, accuracy, null); } else if (config.type === 'regressor') { let mse = 0; currentDataset.forEach(sample => { const prediction = config.predict(sample.inputs, trainedModel); mse += (prediction - sample.target)**2; }); mse = currentDataset.length > 0 ? mse / currentDataset.length : 0; updateMetricsDisplay(false, null, mse); } }
        function updateMetricsDisplay(clear = false, accuracy = null, mse = null) { metricAlgorithmSpan.textContent = ALGORITHM_CONFIG[currentAlgorithm]?.name || "-"; if (clear) { metricAccuracySpan.textContent = "-"; metricMSESpan.textContent = "-"; } else { metricAccuracySpan.textContent = accuracy !== null ? (accuracy*100).toFixed(1)+"%" : "-"; metricMSESpan.textContent = mse !== null ? mse.toFixed(4) : "-"; } }
        function resetPlayground() { updateSliderDisplayValues(); generateDataset(); trainButton.textContent = "Initiate Training"; trainButton.disabled = false; }

        // --- Visualization (Adapted for dark theme colors) ---
        const DB_RESOLUTION = 35;
        function drawDataPoints() {
            const canvas = dataPointsCanvas; const ctx = dpCtx;
            canvas.width = canvas.parentElement.offsetWidth; canvas.height = canvas.parentElement.offsetHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const pointRadius = Math.max(3, Math.min(6, canvas.width/110));
            const isRegression = ALGORITHM_CONFIG[currentAlgorithm]?.type === 'regressor';
            const class0Color = "rgba(0, 200, 255, 0.9)"; // Bright Cyan
            const class1Color = "rgba(220, 50, 220, 0.9)"; // Bright Magenta/Purple
            const regressionColor = "rgba(0, 255, 150, 0.8)"; // Bright Green/Teal

            currentDataset.forEach(p => {
                const x = p.inputs[0] * canvas.width;
                const y_plot_val = p.inputs.length > 1 && !isRegression ? p.inputs[1] : p.target;
                const y = (1 - y_plot_val) * canvas.height;
                ctx.beginPath(); ctx.arc(x, y, pointRadius, 0, 2 * Math.PI);
                if (isRegression) { ctx.fillStyle = regressionColor; }
                else { ctx.fillStyle = p.target === 0 ? class0Color : class1Color; }
                ctx.fill();
                // Optional: add a subtle glow or border to points
                ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth=0.5; ctx.stroke();
            });
        }
        function clearDecisionBoundary() { const canvas = decisionBoundaryCanvas; canvas.width = canvas.parentElement.offsetWidth; canvas.height = canvas.parentElement.offsetHeight; dbCtx.clearRect(0, 0, canvas.width, canvas.height); }
        function drawVisualization() { clearDecisionBoundary(); const config = ALGORITHM_CONFIG[currentAlgorithm]; if (!trainedModel || !config || !config.predict) return; if (config.type === 'classifier') { drawDecisionBoundaryForClassifier(config); } else if (config.type === 'regressor') { drawRegressionLine(config); } }
        function drawDecisionBoundaryForClassifier(config) {
            const canvas = decisionBoundaryCanvas; const ctx = dbCtx;
            const stepX = canvas.width/DB_RESOLUTION; const stepY = canvas.height/DB_RESOLUTION;
            const boundaryClass0Color = "rgba(0, 150, 255, 0.2)"; // Lighter cyan for boundary
            const boundaryClass1Color = "rgba(200, 50, 200, 0.2)"; // Lighter magenta for boundary

            for (let i = 0; i < DB_RESOLUTION; i++) {
                for (let j = 0; j < DB_RESOLUTION; j++) {
                    const normX = (i+0.5)/DB_RESOLUTION; const normY = 1-((j+0.5)/DB_RESOLUTION);
                    const inputsForPrediction = currentDataset[0].inputs.length > 1 ? [normX,normY] : [normX];
                    const prediction = config.predict(inputsForPrediction, trainedModel);
                    ctx.fillStyle = prediction === 0 ? boundaryClass0Color : boundaryClass1Color;
                    ctx.fillRect(i * stepX, j * stepY, stepX, stepY);
                }
            }
        }
        function drawRegressionLine(config) {
            const canvas = decisionBoundaryCanvas; const ctx = dbCtx;
            ctx.beginPath(); const stepX = canvas.width/100;
            for (let i = 0; i <= 100; i++) {
                const normX = i/100; const prediction = config.predict([normX], trainedModel);
                const x_canvas = normX * canvas.width; const y_canvas = (1-prediction)*canvas.height;
                if (i === 0) ctx.moveTo(x_canvas, y_canvas); else ctx.lineTo(x_canvas, y_canvas);
            }
            ctx.strokeStyle = 'var(--accent-success)'; ctx.lineWidth = 3;
            ctx.shadowColor = 'var(--accent-success)'; ctx.shadowBlur = 5; // Add glow to line
            ctx.stroke();
            ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; // Reset shadow
        }
        function updateVisualizationCaption() { const config = ALGORITHM_CONFIG[currentAlgorithm]; if (config.type === 'classifier') { visualizationCaption.textContent = "CLASSIFICATION GRID // Class 0: Cyan Regions, Class 1: Magenta Regions. Dots are data samples."; } else if (config.type === 'regressor') { visualizationCaption.textContent = "REGRESSION ANALYSIS // Teal dots: Data samples. Green line: Learned regression model."; } }
        
        function updateSliderDisplayValues() { numSamplesValueSpan.textContent = numSamplesSlider.value; noiseLevelValueSpan.textContent = parseFloat(noiseLevelSlider.value).toFixed(2); const activeAlgoConfig = ALGORITHM_CONFIG[currentAlgorithm]; if (activeAlgoConfig && activeAlgoConfig.params) { activeAlgoConfig.params.forEach(param => { const el = document.getElementById(param.id); if(el && param.type === 'range') { const valEl = document.getElementById(`${param.id}Value`); if(valEl) valEl.textContent = el.value; } }); } }

        // --- Initial Setup & Event Listeners ---
        algorithmSelect.addEventListener('change', handleAlgorithmChange);
        datasetTypeSelect.addEventListener('change', () => { trainedModel = null; generateDataset(); });
        numSamplesSlider.addEventListener('input', () => { numSamplesValueSpan.textContent = numSamplesSlider.value; /* generateDataset(); */ }); // Consider auto-gen on release
        noiseLevelSlider.addEventListener('input', () => { noiseLevelValueSpan.textContent = parseFloat(noiseLevelSlider.value).toFixed(2); /* generateDataset(); */ });
        generateDataButton.addEventListener('click', generateDataset);
        trainButton.addEventListener('click', trainSelectedModel);
        resetButton.addEventListener('click', resetPlayground);

        let resizeTimeout; window.addEventListener('resize', () => { clearTimeout(resizeTimeout); resizeTimeout = setTimeout(() => { if (currentDataset.length > 0) drawDataPoints(); if (trainedModel) drawVisualization(); }, 200); });

        updateHyperparameterUI(); generateDataset(); updateSliderDisplayValues();
    });
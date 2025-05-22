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

    const dataPointsCanvas = document.getElementById('dataPointsCanvas');
    const dpCtx = dataPointsCanvas.getContext('2d');
    const modelVisCanvas = document.getElementById('modelVisualizationCanvas');
    const modelVisCtx = modelVisCanvas.getContext('2d');
    const visualizationCaption = document.getElementById('visualizationCaption');

    const metricAlgorithmSpan = document.getElementById('metricAlgorithm');
    const metricAccuracySpan = document.getElementById('metricAccuracy');
    const accuracyMetricItem = document.getElementById('accuracyMetricItem');
    const metricMSESpan = document.getElementById('metricMSE');
    const mseMetricItem = document.getElementById('mseMetricItem');
    const metricRMSESpan = document.getElementById('metricRMSE'); 
    const rmseMetricItem = document.getElementById('rmseMetricItem'); 
    const metricMAESpan = document.getElementById('metricMAE');   
    const maeMetricItem = document.getElementById('maeMetricItem');   
    const metricRSquaredSpan = document.getElementById('metricRSquared'); 
    const rSquaredMetricItem = document.getElementById('rSquaredMetricItem'); 

    const infoAlgorithmNameSpan = document.getElementById('infoAlgorithmName');
    const infoAlgorithmDescriptionP = document.getElementById('infoAlgorithmDescription');

    // --- App State ---
    let currentDataset = []; 
    let currentAlgorithmKey = algorithmSelect.value;
    let trainedModel = null; 
    let hyperparameters = {}; // Stores current values of hyperparameters
    window.ALGORITHMS = window.ALGORITHMS || {}; 

    // --- Dataset Generation ---
    function generateDataset() {
        const type = datasetTypeSelect.value;
        const n = parseInt(numSamplesSlider.value);
        const noise = parseFloat(noiseLevelSlider.value);
        currentDataset = [];
        
        function randn_bm() { let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random(); return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v); }
        const rNoise = (scale = 1) => (Math.random() - 0.5) * 2 * noise * scale;

        for (let i = 0; i < n; i++) {
            let x_feat, y_feat, target_val_or_class;
            x_feat = Math.random(); 

            switch (type) {
                case 'circles':
                    const r_circle = Math.random() > 0.5 ? 0.22 : 0.42;
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
                    y_feat = 0.2 + 0.6 * x_feat + rNoise(0.25);
                    currentDataset.push({ inputs: [x_feat], target: y_feat });
                    break;
                case 'sineWave':
                    y_feat = 0.5 + 0.3 * Math.sin(x_feat * 2.5 * Math.PI) + rNoise(0.15);
                    currentDataset.push({ inputs: [x_feat], target: y_feat });
                    break;
                case 'gauss':
                default:
                    const mean1=[0.3,0.3], mean2=[0.7,0.7]; const std_g=0.1;
                    if(i<n/2){x_f=mean1[0]+randn_bm()*std_g+rNoise(0.2); y_f=mean1[1]+randn_bm()*std_g+rNoise(0.2); t=0;}
                    else{x_f=mean2[0]+randn_bm()*std_g+rNoise(0.2); y_f=mean2[1]+randn_bm()*std_g+rNoise(0.2); t=1;}
                    currentDataset.push({inputs:[norm(x_f),norm(y_f)], target:t});
                    break;
            }
        }
        function norm(val){ return Math.max(0.01, Math.min(0.99, val)); }

        trainedModel = null;
        drawDataPoints();
        clearModelVisualization();
        updateMetricsDisplay(true); 
        updateVisualizationCaption();
        // After generating new data, k for k-NN might need adjustment if slider max depends on N
        if (window.ALGORITHMS[currentAlgorithmKey]?.name.includes("k-NN")) {
            adjustKnnMaxK();
        }
    }
    
    // --- UI & Control Logic ---
    function updateHyperparameterUI() {
        hyperparamsContainer.innerHTML = ''; 
        const config = window.ALGORITHMS[currentAlgorithmKey];
        hyperparameters = {}; // Reset current hyperparameter values for the new algorithm

        if (!config) { 
            console.error(`Algorithm config for '${currentAlgorithmKey}' not found.`);
            hyperparamsContainer.innerHTML = `<p style="color: var(--accent-danger);">Error: Config for '${currentAlgorithmKey}' missing.</p>`;
            infoAlgorithmNameSpan.textContent = "Error";
            infoAlgorithmDescriptionP.textContent = "Algorithm not found.";
            return;
        }
        if (!config.params || config.params.length === 0) {
            hyperparamsContainer.innerHTML = '<p style="color: var(--text-secondary);">No configurable hyperparameters for this algorithm.</p>';
        } else {
            config.params.forEach(param => {
                const group = document.createElement('div'); 
                group.className = 'input-group';
                group.id = `group-${param.id}`; // For easier selection if needed

                const label = document.createElement('label'); 
                label.setAttribute('for', param.id);
                const valueSpanId = `${param.id}Value`;
                const isRange = param.type === 'range';
                label.innerHTML = `${param.label}: ${isRange ? `<span id="${valueSpanId}" class="range-value">${param.default}</span>` : ''}`;
                
                let input;
                if (param.type === 'select') {
                    input = document.createElement('select');
                    param.options.forEach(opt => {
                        const optionEl = document.createElement('option');
                        optionEl.value = opt.value;
                        optionEl.textContent = opt.text;
                        input.appendChild(optionEl);
                    });
                } else {
                    input = document.createElement('input');
                    input.type = param.type === 'range' ? 'range' : 'number';
                    if (param.type === 'range' || input.type === 'number') {
                        input.min = param.min; 
                        input.max = param.max; 
                        input.step = param.step;
                    }
                }
                input.id = param.id; 
                input.name = param.id;
                input.value = param.default;
                
                // Initialize hyperparameter with default value
                hyperparameters[param.id] = param.type === 'number' ? parseFloat(param.default) : 
                                           (isRange ? parseInt(param.default) : param.default);

                // Store condition function directly on the group element
                if (param.condition) {
                    group._conditionFn = param.condition;
                }

                input.addEventListener('input', (e) => {
                    const val = param.type === 'number' ? parseFloat(e.target.value) : 
                               (isRange ? parseInt(e.target.value) : e.target.value);
                    hyperparameters[param.id] = val; // Update global hyperparameters object
                    if (isRange) {
                        const valSpan = document.getElementById(valueSpanId);
                        if (valSpan) valSpan.textContent = e.target.value;
                    }
                    
                    trainedModel = null; clearModelVisualization(); updateMetricsDisplay(true);
                    checkConditionalParamsVisibility(); // Check if this change affects other params
                });
                group.appendChild(label); group.appendChild(input); hyperparamsContainer.appendChild(group);
            });
        }
        infoAlgorithmNameSpan.textContent = config.name;
        infoAlgorithmDescriptionP.textContent = config.description;
        metricAlgorithmSpan.textContent = config.name;

        const isRegressor = config.type === 'regressor';
        mseMetricItem.style.display = isRegressor ? 'flex' : 'none';
        rmseMetricItem.style.display = isRegressor ? 'flex' : 'none';
        maeMetricItem.style.display = isRegressor ? 'flex' : 'none';
        rSquaredMetricItem.style.display = isRegressor ? 'flex' : 'none';
        accuracyMetricItem.style.display = !isRegressor ? 'flex' : 'none';
        
        updateVisualizationCaption();
        updateDatasetOptions(config.type);
        checkConditionalParamsVisibility(); // Initial check for conditional params
    }

    function checkConditionalParamsVisibility() {
        const paramGroups = hyperparamsContainer.querySelectorAll('.input-group');
        paramGroups.forEach(group => {
            if (group._conditionFn) { // Check if condition function is stored
                group.style.display = group._conditionFn(hyperparameters) ? 'block' : 'none';
            } else {
                group.style.display = 'block'; // Default to visible if no condition
            }
        });
    }

    function updateDatasetOptions(algorithmType) {
        const isRegression = algorithmType === 'regressor';
        let firstEnabledOptionValue = null;

        Array.from(datasetTypeSelect.options).forEach(option => {
            const isRegressionDataset = option.value.includes("Trend") || option.value.includes("Wave");
            option.disabled = isRegression ? !isRegressionDataset : isRegressionDataset;
            if (!option.disabled && !firstEnabledOptionValue) {
                firstEnabledOptionValue = option.value;
            }
        });

        if (datasetTypeSelect.options[datasetTypeSelect.selectedIndex].disabled) {
            datasetTypeSelect.value = firstEnabledOptionValue || (isRegression ? 'linearTrend' : 'gauss');
            generateDataset(); 
        }
    }
    
    function handleAlgorithmChange() {
        currentAlgorithmKey = algorithmSelect.value;
        const algoConfig = window.ALGORITHMS[currentAlgorithmKey];
        if (!algoConfig) {
            console.error(`Algorithm ${currentAlgorithmKey} not loaded or defined.`);
            hyperparamsContainer.innerHTML = `<p style="color:var(--accent-danger);">Error: Algorithm '${currentAlgorithmKey}' definition not found. Is the script loaded?</p>`;
            infoAlgorithmNameSpan.textContent = "Error";
            infoAlgorithmDescriptionP.textContent = "Algorithm not found.";
            metricAccuracySpan.textContent = "-"; metricMSESpan.textContent = "-";
            metricRMSESpan.textContent = "-"; metricMAESpan.textContent = "-";
            metricRSquaredSpan.textContent = "-";
            return;
        }
        updateHyperparameterUI(); 
        trainedModel = null; clearModelVisualization(); updateMetricsDisplay(true);
    }

    function trainSelectedModel() {
        if (currentDataset.length === 0) { alert("DATA MATRIX empty. Please generate data first."); return; }
        const config = window.ALGORITHMS[currentAlgorithmKey];
        if (config && config.train) {
            trainButton.disabled = true;
            trainButton.textContent = "TRAINING...";
            
            setTimeout(() => {
                try {
                    trainedModel = config.train(currentDataset, { ...hyperparameters });
                    evaluateModel(); 
                    drawModelVisualization();
                } catch (error) {
                    console.error(`Training error for ${currentAlgorithmKey}:`, error);
                    alert(`An error occurred during training for ${config.name}. Check console.`);
                } finally {
                    trainButton.disabled = false;
                    trainButton.textContent = "RE-INITIATE TRAINING";
                }
            }, 20); 
        } else { alert(`Algorithm ${currentAlgorithmKey} does not have a 'train' method defined.`); }
    }

    function evaluateModel() {
        if (!trainedModel || currentDataset.length === 0) { updateMetricsDisplay(true); return; }
        const config = window.ALGORITHMS[currentAlgorithmKey];
        if (!config || !config.predict) {
            console.warn(`Predict function for ${currentAlgorithmKey} not found.`);
            updateMetricsDisplay(true); return;
        }

        let accuracy = null, mse = null, rmse = null, mae = null, rSquared = null;

        if (config.type === 'classifier') {
            let correct = 0;
            currentDataset.forEach(sample => {
                const prediction = config.predict(sample.inputs, trainedModel);
                if (prediction === sample.target) correct++;
            });
            accuracy = currentDataset.length > 0 ? (correct / currentDataset.length) : 0;
        } else if (config.type === 'regressor') {
            let sumSquaredError = 0; let sumAbsoluteError = 0; let sumTargets = 0;
            const predictions = []; const targets = [];

            currentDataset.forEach(sample => {
                const prediction = config.predict(sample.inputs, trainedModel);
                predictions.push(prediction); targets.push(sample.target);
                const error = prediction - sample.target;
                sumSquaredError += error * error; sumAbsoluteError += Math.abs(error);
                sumTargets += sample.target;
            });

            if (currentDataset.length > 0) {
                mse = sumSquaredError / currentDataset.length; rmse = Math.sqrt(mse);
                mae = sumAbsoluteError / currentDataset.length;
                const meanTarget = sumTargets / currentDataset.length;
                let totalSumOfSquares = 0;
                targets.forEach(target => { totalSumOfSquares += (target - meanTarget) ** 2; });
                rSquared = totalSumOfSquares === 0 ? (sumSquaredError === 0 ? 1 : 0) : 1 - (sumSquaredError / totalSumOfSquares); 
            }
        }
        updateMetricsDisplay(false, accuracy, mse, rmse, mae, rSquared);
    }

    function updateMetricsDisplay(clear = false, accuracy = null, mse = null, rmse = null, mae = null, rSquared = null) {
        const config = window.ALGORITHMS[currentAlgorithmKey];
        metricAlgorithmSpan.textContent = config ? config.name : "-";
        if (clear) {
            metricAccuracySpan.textContent = "-"; metricMSESpan.textContent = "-";
            metricRMSESpan.textContent = "-"; metricMAESpan.textContent = "-";
            metricRSquaredSpan.textContent = "-";
        } else {
            metricAccuracySpan.textContent = accuracy !== null ? (accuracy * 100).toFixed(1) + "%" : "-";
            metricMSESpan.textContent = mse !== null ? mse.toFixed(4) : "-";
            metricRMSESpan.textContent = rmse !== null ? rmse.toFixed(4) : "-";
            metricMAESpan.textContent = mae !== null ? mae.toFixed(4) : "-";
            metricRSquaredSpan.textContent = rSquared !== null ? rSquared.toFixed(3) : "-";
        }
    }
    
    function resetPlayground() {
        updateSliderDisplayValues(); generateDataset(); 
        trainButton.textContent = "Initiate Training"; trainButton.disabled = false;
    }

    // --- Visualization --- 
    const DB_RESOLUTION = 35;
    function drawDataPoints() {
        const canvas = dataPointsCanvas; const ctx = dpCtx;
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth; canvas.height = container.offsetHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if(currentDataset.length === 0) return;
        const pointRadius = Math.max(3, Math.min(6, canvas.width/110));
        const algoConfig = window.ALGORITHMS[currentAlgorithmKey];
        const isRegression = algoConfig && algoConfig.type === 'regressor';
        const class0Color = "rgba(0, 200, 255, 0.9)"; const class1Color = "rgba(220, 50, 220, 0.9)";
        const regressionColor = "rgba(0, 255, 150, 0.8)";
        currentDataset.forEach(p => {
            const x_canvas = p.inputs[0] * canvas.width;
            let y_canvas_val;
            if (isRegression || p.inputs.length === 1) { y_canvas_val = p.target; }
            else { y_canvas_val = p.inputs[1]; }
            const y_canvas = (1 - y_canvas_val) * canvas.height;
            ctx.beginPath(); ctx.arc(x_canvas, y_canvas, pointRadius, 0, 2 * Math.PI);
            if (isRegression) { ctx.fillStyle = regressionColor; }
            else { ctx.fillStyle = p.target === 0 ? class0Color : class1Color; }
            ctx.fill(); ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.lineWidth=0.5; ctx.stroke();
        });
    }
    function clearModelVisualization() {
        const canvas = modelVisCanvas; const ctx = modelVisCtx;
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth; canvas.height = container.offsetHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function drawModelVisualization() { 
        clearModelVisualization(); const config = window.ALGORITHMS[currentAlgorithmKey];
        if (!trainedModel || !config || !config.predict) return;
        if (config.type === 'classifier') { drawDecisionBoundaryForClassifier(config); }
        else if (config.type === 'regressor') { drawRegressionLine(config); }
    }
    function drawDecisionBoundaryForClassifier(config) {
        const canvas = modelVisCanvas; const ctx = modelVisCtx;
        const stepX = canvas.width/DB_RESOLUTION; const stepY = canvas.height/DB_RESOLUTION;
        const boundaryClass0Color = "rgba(0, 150, 255, 0.18)"; const boundaryClass1Color = "rgba(200, 50, 200, 0.18)";
        for (let i = 0; i < DB_RESOLUTION; i++) { for (let j = 0; j < DB_RESOLUTION; j++) {
            const normX=(i+0.5)/DB_RESOLUTION; const normY=1-((j+0.5)/DB_RESOLUTION);
            const inputsForPred = (currentDataset.length > 0 && currentDataset[0].inputs.length > 1) ? [normX,normY] : [normX];
            const pred = config.predict(inputsForPred, trainedModel);
            ctx.fillStyle = pred === 0 ? boundaryClass0Color : boundaryClass1Color;
            ctx.fillRect(i*stepX, j*stepY, stepX, stepY);
        }}
    }
    function drawRegressionLine(config) {
        const canvas = modelVisCanvas; const ctx = modelVisCtx;
        ctx.beginPath(); const linePoints = 100;
        for (let i = 0; i <= linePoints; i++) {
            const normX = i/linePoints; const pred = config.predict([normX], trainedModel);
            const x_c = normX*canvas.width; const y_c = (1-pred)*canvas.height;
            if (i===0) ctx.moveTo(x_c, y_c); else ctx.lineTo(x_c, y_c);
        }
        ctx.strokeStyle = 'var(--accent-success)'; ctx.lineWidth = 3;
        ctx.shadowColor = 'var(--accent-success)'; ctx.shadowBlur = 6; ctx.stroke();
        ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; 
    }
    function updateVisualizationCaption() {
        const config = window.ALGORITHMS[currentAlgorithmKey]; if(!config) return;
        if (config.type === 'classifier') { visualizationCaption.textContent = "CLASSIFICATION GRID // Class 0: Cyan Regions, Class 1: Magenta Regions. Dots are data samples."; }
        else if (config.type === 'regressor') { visualizationCaption.textContent = "REGRESSION ANALYSIS // Teal dots: Data samples. Green line: Learned regression model."; }
    }
    function updateSliderDisplayValues() {
        numSamplesValueSpan.textContent = numSamplesSlider.value;
        noiseLevelValueSpan.textContent = parseFloat(noiseLevelSlider.value).toFixed(2);
        const activeAlgoConfig = window.ALGORITHMS[currentAlgorithmKey];
        if (activeAlgoConfig && activeAlgoConfig.params) { activeAlgoConfig.params.forEach(param => {
            const el = document.getElementById(param.id);
            if(el && param.type === 'range') { const valEl = document.getElementById(`${param.id}Value`); if(valEl) valEl.textContent = el.value; }
        });}
    }
    function adjustKnnMaxK() { // Specific adjustment for k-NN
        if (currentAlgorithmKey === 'knn' && currentDataset.length > 0) {
            const kSlider = document.getElementById('knnK');
            const kValueSpan = document.getElementById('knnKValue');
            if (kSlider && kValueSpan) {
                const configKParam = window.ALGORITHMS.knn.params.find(p => p.id === 'knnK');
                const UIMaxK = configKParam ? parseInt(configKParam.max) : 25; // Get max from config
                const actualMaxK = Math.min(UIMaxK, currentDataset.length > 0 ? currentDataset.length : UIMaxK);
                
                kSlider.max = actualMaxK; // Update slider's actual max attribute
                if (parseInt(kSlider.value) > actualMaxK) {
                    kSlider.value = actualMaxK;
                    hyperparameters['knnK'] = actualMaxK;
                    kValueSpan.textContent = actualMaxK;
                }
            }
        }
    }


    // --- Initial Setup & Event Listeners ---
    algorithmSelect.addEventListener('change', handleAlgorithmChange);
    datasetTypeSelect.addEventListener('change', () => { trainedModel = null; generateDataset(); });
    numSamplesSlider.addEventListener('input', () => { numSamplesValueSpan.textContent = numSamplesSlider.value; adjustKnnMaxK();/* Can call generateDataset() on 'change' event instead */});
    noiseLevelSlider.addEventListener('input', () => { noiseLevelValueSpan.textContent = parseFloat(noiseLevelSlider.value).toFixed(2); /* Can call generateDataset() on 'change' event */});
    generateDataButton.addEventListener('click', generateDataset);
    trainButton.addEventListener('click', trainSelectedModel);
    resetButton.addEventListener('click', resetPlayground);

    let resizeTimeout; window.addEventListener('resize', () => { clearTimeout(resizeTimeout); resizeTimeout = setTimeout(() => {
        const container = dataPointsCanvas.parentElement;
        if(container) { // Ensure container exists
            dataPointsCanvas.width = container.offsetWidth; dataPointsCanvas.height = container.offsetHeight;
            modelVisCanvas.width = container.offsetWidth; modelVisCanvas.height = container.offsetHeight;
            if (currentDataset.length > 0) drawDataPoints(); if (trainedModel) drawModelVisualization();
        }
    }, 200);});

    // Initial calls
    handleAlgorithmChange(); 
    generateDataset();      
    updateSliderDisplayValues();
});

// ============================================
// CROP YIELD AI - NETLIFY BROWSER INFERENCE
// No Flask / Railway / Render API required
// ============================================

const MODEL_PATH = "./models/crop_yield_pipeline.onnx";
const OPTIONS_PATH = "./options.json";
const HISTORY_KEY = "cropYieldPredictionHistoryV1";

const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");
const compareBtn = document.getElementById("compareBtn");
const yearInput = document.getElementById("year");

let modelSession = null;
let datasetOptions = null;

// --------------------------------------------
// STATUS
// --------------------------------------------

function setApiStatus(online, message) {
    const badge = document.getElementById("apiStatus");
    if (!badge) return;

    badge.textContent = message;
    badge.classList.toggle("is-online", online);
    badge.classList.toggle("is-offline", !online);
}

function showError(message) {
    result.replaceChildren();

    const panel = document.createElement("div");
    panel.className = "error-message";
    panel.setAttribute("role", "alert");
    panel.textContent = message;

    result.appendChild(panel);
}

// --------------------------------------------
// LOAD DATASET OPTIONS
// --------------------------------------------

function setSelectOptions(id, values, placeholder) {
    const select = document.getElementById(id);

    select.replaceChildren(new Option(placeholder, ""));

    values.forEach(value => {
        select.add(new Option(value, value));
    });
}

async function loadOptions() {
    const response = await fetch(OPTIONS_PATH);

    if (!response.ok) {
        throw new Error("Dataset options could not be loaded.");
    }

    const data = await response.json();
    datasetOptions = data;

    setSelectOptions("state", data.states, "Select State");
    setSelectOptions("crop", data.crops, "Select Crop");
    setSelectOptions("season", data.seasons, "Select Season");
    setSelectOptions(
        "compareCrops",
        data.crops,
        "Select crops to compare"
    );

    yearInput.min = data.year_min;
    yearInput.max = data.year_max;
    yearInput.value = data.year_max;

    const metrics = {
        datasetRecords: Number(data.dataset_rows).toLocaleString(),
        datasetCrops: data.crops.length.toLocaleString(),
        datasetStates: data.states.length.toLocaleString(),
        datasetYears: `${data.year_min}–${data.year_max}`
    };

    Object.entries(metrics).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });

    const qualityMetric =
        document.getElementById("datasetConsistency");

    if (qualityMetric && data.data_quality) {
        qualityMetric.textContent =
            `${Number(data.data_quality.match_within_5pct_pct).toFixed(1)}%`;
    }

    return data;
}

// --------------------------------------------
// LOAD ONNX MODEL
// --------------------------------------------

async function loadModel() {
    if (!window.ort) {
        throw new Error("ONNX Runtime could not be loaded. Check your internet connection.");
    }

    ort.env.wasm.numThreads = 1;

    ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

    modelSession = await ort.InferenceSession.create(
        MODEL_PATH,
        {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all"
        }
    );

    console.log("ONNX model loaded successfully.");
    console.log("Model inputs:", modelSession.inputNames);
    console.log("Model outputs:", modelSession.outputNames);

    return modelSession;
}

// --------------------------------------------
// INPUT HANDLING
// --------------------------------------------

function readForm() {
    return {
        Year: Number(document.getElementById("year").value),
        State: document.getElementById("state").value,
        Crop: document.getElementById("crop").value,
        Season: document.getElementById("season").value,
        Area: Number(document.getElementById("area").value),
        Area_Unit: document.getElementById("areaUnit").value,
        Annual_Rainfall: Number(
            document.getElementById("rainfall").value
        ),
        Fertilizer: Number(
            document.getElementById("fertilizer").value
        ),
        Pesticide: Number(
            document.getElementById("pesticide").value
        )
    };
}

function validateInput(data) {
    if (!datasetOptions) {
        return "Dataset is not ready.";
    }

    if (
        !Number.isInteger(data.Year) ||
        data.Year < datasetOptions.year_min ||
        data.Year > datasetOptions.year_max
    ) {
        return `Year must be between ${datasetOptions.year_min} and ${datasetOptions.year_max}.`;
    }

    if (!datasetOptions.states.includes(data.State)) {
        return "Please select a valid state.";
    }

    if (!datasetOptions.crops.includes(data.Crop)) {
        return "Please select a valid crop.";
    }

    if (!datasetOptions.seasons.includes(data.Season)) {
        return "Please select a valid season.";
    }

    if (!Number.isFinite(data.Area) || data.Area <= 0) {
        return "Area must be greater than zero.";
    }

    if (!["hectare", "acre"].includes(data.Area_Unit)) {
        return "Select hectares or acres.";
    }

    if (
        !Number.isFinite(data.Annual_Rainfall) ||
        data.Annual_Rainfall < 0
    ) {
        return "Rainfall cannot be negative.";
    }

    if (!Number.isFinite(data.Fertilizer) || data.Fertilizer < 0) {
        return "Fertilizer cannot be negative.";
    }

    if (!Number.isFinite(data.Pesticide) || data.Pesticide < 0) {
        return "Pesticide cannot be negative.";
    }

    return null;
}

// --------------------------------------------
// UNIT CONVERSION
// --------------------------------------------

function areaInHectares(data) {
    if (data.Area_Unit === "acre") {
        return data.Area * 0.40468564224;
    }

    return data.Area;
}

// --------------------------------------------
// ONNX PREDICTION
// --------------------------------------------

async function predictBatch(inputs) {
    if (!modelSession) {
        throw new Error("ML model is still loading. Please wait.");
    }

    const count = inputs.length;

    const numericFields = [
        "Year",
        "Area",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ];

    const categoricalFields = [
        "State",
        "Crop",
        "Season"
    ];

    const feed = {};

    numericFields.forEach(field => {
        const values = inputs.map(item => {
            return field === "Area"
                ? areaInHectares(item)
                : item[field];
        });

        feed[field] = new ort.Tensor(
            "float32",
            Float32Array.from(values),
            [count, 1]
        );
    });

    categoricalFields.forEach(field => {
        const values = inputs.map(item => String(item[field]));

        feed[field] = new ort.Tensor(
            "string",
            values,
            [count, 1]
        );
    });

    const outputs = await modelSession.run(feed);

    const outputName = modelSession.outputNames[0];
    const predictions = Array.from(outputs[outputName].data)
        .map(Number);

    if (
        predictions.length !== count ||
        predictions.some(value => !Number.isFinite(value) || value < 0)
    ) {
        throw new Error("The ML model returned invalid predictions.");
    }

    return predictions;
}

// --------------------------------------------
// YIELD UNIT INFORMATION
// --------------------------------------------

function getYieldUnit(crop) {
    if (String(crop).trim().toLowerCase() === "coconut") {
        return {
            unit: "Dataset units (physical unit unverified)",
            note: "Coconut target values have an unverified physical unit. Treat this as a dataset-scale estimate only."
        };
    }

    return {
        unit: "Metric tonnes/hectare (t/ha)",
        note: "Estimate based on the dataset's Yield target. It is a statistical estimate, not a guaranteed harvest."
    };
}

// --------------------------------------------
// DISPLAY PREDICTION
// --------------------------------------------

function showPrediction(value, inputs) {
    const units = getYieldUnit(inputs.Crop);

    result.innerHTML = `
        <div class="result-circle">
            <div class="plant-icon">🌱</div>
            <div class="yield-value">${value.toFixed(2)}</div>
            <div class="yield-unit">${units.unit}</div>
        </div>

        <h3>Predicted Crop Yield</h3>

        <div class="result-info">
            <span>💡</span>
            <p>${units.note}</p>
        </div>
    `;

    const details = document.createElement("div");
    details.className = "prediction-details";

    const lines = [
        `Area: ${areaInHectares(inputs).toFixed(3)} hectares`,
        `Input area: ${inputs.Area} ${inputs.Area_Unit}`,
        `Fertilizer: ${inputs.Fertilizer.toLocaleString()} kg total`,
        `Pesticide: ${inputs.Pesticide.toLocaleString()} kg total`,
        `Annual rainfall: ${inputs.Annual_Rainfall.toLocaleString()} mm`
    ];

    lines.forEach(line => {
        const p = document.createElement("p");
        p.textContent = line;
        details.appendChild(p);
    });

    result.appendChild(details);

    const caveat = document.createElement("p");
    caveat.className = "production-caveat";
    caveat.textContent =
        "Total production is not calculated because source Yield and Production / Area values do not consistently agree.";

    result.appendChild(caveat);

    if (inputs.Year >= 2021) {
        const warning = document.createElement("div");
        warning.className = "prediction-warning";
        warning.textContent =
            "The dataset describes 2021–2026 records as trend-extended or gap-filled. This is not a real-time weather forecast.";
        result.appendChild(warning);
    }

    const quality = datasetOptions.data_quality?.match_within_5pct_pct;

    if (Number.isFinite(Number(quality))) {
        const warning = document.createElement("div");
        warning.className = "prediction-warning";
        warning.textContent =
            `Data-quality audit: ${quality}% of records reconcile Yield with Production / Area within 5%.`;
        result.appendChild(warning);
    }

    appendWarning(
        result,
        "This is a state-level statistical estimate, not a farm-specific guarantee or fertilizer/pesticide recommendation."
    );
}

function appendWarning(container, message) {
    const item = document.createElement("div");
    item.className = "prediction-warning";
    item.textContent = message;
    container.appendChild(item);
}

// --------------------------------------------
// HISTORY
// --------------------------------------------

function getHistory() {
    try {
        const items = JSON.parse(
            localStorage.getItem(HISTORY_KEY) || "[]"
        );

        return Array.isArray(items) ? items.slice(0, 8) : [];
    } catch (_) {
        return [];
    }
}

function renderHistory() {
    const container = document.getElementById("predictionHistory");
    if (!container) return;

    const entries = getHistory();
    container.replaceChildren();

    if (!entries.length) {
        const empty = document.createElement("p");
        empty.className = "history-empty";
        empty.textContent =
            "Your recent predictions will appear here on this device.";
        container.appendChild(empty);
        return;
    }

    entries.forEach(entry => {
        const row = document.createElement("article");
        row.className = "history-item";

        const detail = document.createElement("div");

        const title = document.createElement("strong");
        title.textContent = entry.crop;

        const meta = document.createElement("small");
        meta.textContent =
            `${entry.state} · ${entry.season} · ${entry.year}`;

        detail.append(title, meta);

        const amount = document.createElement("b");
        amount.textContent =
            `${Number(entry.yield).toFixed(2)} ${entry.unit || ""}`;

        row.append(detail, amount);
        container.appendChild(row);
    });
}

function saveHistory(data, prediction) {
    const history = getHistory();

    history.unshift({
        crop: data.Crop,
        state: data.State,
        season: data.Season,
        year: data.Year,
        yield: prediction,
        unit: getYieldUnit(data.Crop).unit
    });

    try {
        localStorage.setItem(
            HISTORY_KEY,
            JSON.stringify(history.slice(0, 8))
        );
    } catch (_) {
        // Prediction still succeeds if local storage is unavailable.
    }

    renderHistory();
}

// --------------------------------------------
// PREDICTION FORM
// --------------------------------------------

form.addEventListener("submit", async event => {
    event.preventDefault();

    const data = readForm();
    const validationError = validateInput(data);

    if (validationError) {
        showError(validationError);
        return;
    }

    predictBtn.disabled = true;
    predictBtn.classList.add("loading");
    predictBtn.textContent = "Predicting…";

    result.innerHTML =
        '<div class="result-info"><p>Running the trained model in your browser…</p></div>';

    try {
        const predictions = await predictBatch([data]);
        const value = predictions[0];

        showPrediction(value, data);
        saveHistory(data, value);

    } catch (error) {
        console.error("Prediction error:", error);
        showError(
            error.message || "Prediction failed. Please try again."
        );
    } finally {
        predictBtn.disabled = false;
        predictBtn.classList.remove("loading");
        predictBtn.innerHTML =
            "<span>⌁</span><span>Predict Yield</span>";
    }
});

// --------------------------------------------
// CROP COMPARISON
// --------------------------------------------

compareBtn.addEventListener("click", async () => {
    const crops = Array.from(
        document.getElementById("compareCrops").selectedOptions
    ).map(option => option.value);

    if (crops.length < 2 || crops.length > 8) {
        showError("Select between 2 and 8 crops to compare.");
        return;
    }

    const inputs = readForm();
    const validationError = validateInput(inputs);

    if (validationError) {
        showError(validationError);
        return;
    }

    if (crops.some(crop => crop.toLowerCase() === "coconut")) {
        showError(
            "Coconut cannot be ranked with other crops because its physical yield unit is unverified."
        );
        return;
    }

    compareBtn.disabled = true;
    compareBtn.textContent = "Comparing crops…";

    result.innerHTML =
        '<div class="result-info"><p>Comparing crops using the same farm conditions…</p></div>';

    try {
        const batch = crops.map(crop => ({
            ...inputs,
            Crop: crop
        }));

        const values = await predictBatch(batch);

        const predictions = crops.map((crop, index) => ({
            crop,
            predicted_yield: values[index],
            yield_unit: getYieldUnit(crop).unit
        }));

        predictions.sort(
            (a, b) => b.predicted_yield - a.predicted_yield
        );

        result.replaceChildren();

        const heading = document.createElement("h3");
        heading.textContent = "Crop yield comparison";
        result.appendChild(heading);

        const caption = document.createElement("p");
        caption.className = "comparison-note";
        caption.textContent =
            "Ranked model estimates using identical state, season, year and farm inputs. These are not guaranteed harvest results.";

        result.appendChild(caption);

        predictions.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "comparison-row";

            const crop = document.createElement("span");
            crop.textContent = `${index + 1}. ${item.crop}`;

            const prediction = document.createElement("strong");
            prediction.textContent =
                `${item.predicted_yield.toFixed(2)} ${item.yield_unit}`;

            row.append(crop, prediction);
            result.appendChild(row);
        });

        const note = document.createElement("div");
        note.className = "result-info comparison-winner";
        note.textContent =
            `Highest model estimate: ${predictions[0].crop}. This is not an agronomic recommendation.`;

        result.appendChild(note);

    } catch (error) {
        console.error("Comparison error:", error);
        showError(error.message || "Crop comparison failed.");
    } finally {
        compareBtn.disabled = false;
        compareBtn.textContent = "Compare Selected Crops";
    }
});

// --------------------------------------------
// CLEAR HISTORY
// --------------------------------------------

document.getElementById("clearHistoryBtn")
    .addEventListener("click", () => {
        try {
            localStorage.removeItem(HISTORY_KEY);
        } catch (_) {}

        renderHistory();
    });

// --------------------------------------------
// INITIAL SCREEN
// --------------------------------------------

function showInitialResult() {
    result.innerHTML = `
        <div class="result-circle">
            <div class="plant-icon">🌱</div>
            <div class="yield-value">—</div>
            <div class="yield-unit">Predicted Yield</div>
        </div>

        <h3>Predicted Crop Yield</h3>

        <div class="result-info">
            <span>💡</span>
            <p>
                Enter your farm parameters and predict a yield,
                or compare multiple crops under identical conditions.
            </p>
        </div>
    `;
}

// --------------------------------------------
// START APPLICATION
// --------------------------------------------

async function initializeApp() {
    showInitialResult();
    renderHistory();

    setApiStatus(false, "Loading browser ML model…");

    try {
        await Promise.all([
            loadOptions(),
            loadModel()
        ]);

        setApiStatus(
            true,
            "Browser ML Model Ready"
        );

        console.log("Crop Yield AI initialized successfully.");

    } catch (error) {
        console.error("Application initialization failed:", error);

        setApiStatus(
            false,
            "ML Model unavailable"
        );

        showError(
            error.message ||
            "Could not initialize the prediction system."
        );
    }
}

initializeApp();

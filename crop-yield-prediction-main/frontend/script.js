// Crop Yield Intelligence — frontend application
// Configure a different backend by setting window.CROP_YIELD_API_URL before this file.
const API_BASE_URL = (window.CROP_YIELD_API_URL ||
    "https://crop-yield-prediction-api-cizc.onrender.com").replace(/\/$/, "");

const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");
const compareBtn = document.getElementById("compareBtn");
const yearInput = document.getElementById("year");
const HISTORY_KEY = "cropYieldPredictionHistoryV1";

async function fetchJson(path, options = {}, timeoutMs = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(`${API_BASE_URL}${path}`, {
            ...options,
            signal: controller.signal,
            headers: { Accept: "application/json", ...(options.headers || {}) }
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(body.error || `Request failed (${response.status}).`);
        }
        return body;
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("The server took too long to respond. Please try again.");
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

function setApiStatus(online, message) {
    const badge = document.getElementById("apiStatus");
    if (!badge) return;
    badge.textContent = message;
    badge.classList.toggle("is-online", online);
    badge.classList.toggle("is-offline", !online);
}

function setSelectOptions(id, values, placeholder) {
    const select = document.getElementById(id);
    select.replaceChildren(new Option(placeholder, ""));
    values.forEach(value => select.add(new Option(value, value)));
}

async function loadOptions() {
    try {
        const data = await fetchJson("/options", {}, 20000);
        setSelectOptions("state", data.states || [], "Select State");
        setSelectOptions("crop", data.crops || [], "Select Crop");
        setSelectOptions("season", data.seasons || [], "Select Season");
        setSelectOptions("compareCrops", data.crops || [], "Select crops to compare");

        if (Number.isInteger(data.year_min) && Number.isInteger(data.year_max)) {
            yearInput.min = String(data.year_min);
            yearInput.max = String(data.year_max);
            yearInput.value = String(data.year_max);
        }

        const metrics = {
            datasetRecords: Number(data.dataset_rows || 0).toLocaleString(),
            datasetCrops: (data.crops || []).length.toLocaleString(),
            datasetStates: (data.states || []).length.toLocaleString(),
            datasetYears: `${data.year_min}–${data.year_max}`
        };
        const qualityMetric = document.getElementById("datasetConsistency");
        if (qualityMetric && data.data_quality) {
            qualityMetric.textContent = `${Number(data.data_quality.match_within_5pct_pct).toFixed(1)}%`;
        }
        Object.entries(metrics).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
        return true;
    } catch (error) {
        setApiStatus(false, "Dataset options unavailable");
        ["state", "crop", "season", "compareCrops"].forEach(id => {
            const element = document.getElementById(id);
            if (element) element.disabled = true;
        });
        showError(error.message);
        return false;
    }
}

function readForm() {
    return {
        Year: Number(document.getElementById("year").value),
        State: document.getElementById("state").value,
        Crop: document.getElementById("crop").value,
        Season: document.getElementById("season").value,
        Area: Number(document.getElementById("area").value),
        Area_Unit: document.getElementById("areaUnit").value,
        Annual_Rainfall: Number(document.getElementById("rainfall").value),
        Fertilizer: Number(document.getElementById("fertilizer").value),
        Pesticide: Number(document.getElementById("pesticide").value)
    };
}

function validateInput(data) {
    const minYear = Number(yearInput.min || 2000);
    const maxYear = Number(yearInput.max || 2026);
    if (!Number.isInteger(data.Year) || data.Year < minYear || data.Year > maxYear) {
        return `Year must be a whole number between ${minYear} and ${maxYear}.`;
    }
    if (!data.State) return "Please select a state.";
    if (!data.Crop) return "Please select a crop.";
    if (!data.Season) return "Please select a season.";
    if (!Number.isFinite(data.Area) || data.Area <= 0) return "Cultivated area must be greater than zero.";
    if (!["hectare", "acre"].includes(data.Area_Unit)) return "Choose hectares or acres for cultivated area.";
    if (!Number.isFinite(data.Annual_Rainfall) || data.Annual_Rainfall < 0) return "Annual rainfall cannot be negative.";
    if (!Number.isFinite(data.Fertilizer) || data.Fertilizer < 0) return "Total fertilizer (kg) cannot be negative.";
    if (!Number.isFinite(data.Pesticide) || data.Pesticide < 0) return "Total pesticide (kg) cannot be negative.";
    return null;
}

function showError(message) {
    result.replaceChildren();
    const panel = document.createElement("div");
    panel.className = "error-message";
    panel.setAttribute("role", "alert");
    panel.textContent = message;
    result.appendChild(panel);
}

function appendWarning(container, message) {
    const item = document.createElement("div");
    item.className = "prediction-warning";
    item.textContent = message;
    container.appendChild(item);
}

function showPrediction(payload, inputs) {
    const value = Number(payload.predicted_yield);
    if (!Number.isFinite(value)) throw new Error("The prediction service returned an invalid result.");

    result.innerHTML = `
        <div class="result-circle">
            <div class="plant-icon">🌱</div>
            <div class="yield-value">${value.toFixed(2)}</div>
            <div class="yield-unit">${payload.yield_unit || "Dataset yield units"}</div>
        </div>
        <h3>Predicted Crop Yield</h3>
        <div class="result-info">
            <span>💡</span>
            <p>${payload.yield_unit_note || "Estimate generated by the trained machine learning model."}</p>
        </div>
    `;

    const details = document.createElement("div");
    details.className = "prediction-details";
    const area = document.createElement("p");
    area.textContent = `Model area: ${Number(payload.area_hectares).toFixed(3)} hectares (converted from ${inputs.Area} ${inputs.Area_Unit}).`;
    const fertilizer = document.createElement("p");
    fertilizer.textContent = `Fertilizer input: ${Number(payload.fertilizer_kg).toLocaleString()} kg total.`;
    const pesticide = document.createElement("p");
    pesticide.textContent = `Pesticide input: ${Number(payload.pesticide_kg).toLocaleString()} kg total.`;
    const rainfall = document.createElement("p");
    rainfall.textContent = `Annual rainfall input: ${Number(payload.annual_rainfall_mm).toLocaleString()} mm.`;
    details.append(area, fertilizer, pesticide, rainfall);
    result.appendChild(details);

    const noProduction = document.createElement("p");
    noProduction.className = "production-caveat";
    noProduction.textContent = payload.total_production_note ||
        "Total production is not calculated because source yield/production units need further validation.";
    result.appendChild(noProduction);

    (payload.warnings || []).forEach(message => appendWarning(result, message));
}

function getHistory() {
    try {
        const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
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
        empty.textContent = "Your recent predictions will appear here on this device.";
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
        meta.textContent = `${entry.state} · ${entry.season} · ${entry.year}`;
        detail.append(title, meta);
        const amount = document.createElement("b");
        amount.textContent = `${Number(entry.yield).toFixed(2)} ${entry.unit || ""}`.trim();
        row.append(detail, amount);
        container.appendChild(row);
    });
}

function saveHistory(data, prediction) {
    const history = getHistory();
    history.unshift({
        crop: data.Crop, state: data.State, season: data.Season,
        year: data.Year, yield: prediction, unit: data.yield_unit || ""
    });
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 8)));
    } catch (_) {
        // Local storage may be unavailable in private browsing; prediction still succeeds.
    }
    renderHistory();
}

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
    result.innerHTML = '<div class="result-info"><p>Running the trained model…</p></div>';

    try {
        const payload = await fetchJson("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        }, 30000);
        const value = Number(payload.predicted_yield);
        if (!Number.isFinite(value)) throw new Error("The prediction service returned an invalid result.");
        showPrediction(payload, data);
        saveHistory({ ...data, yield_unit: payload.yield_unit }, value);
    } catch (error) {
        showError(error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.classList.remove("loading");
        predictBtn.innerHTML = "<span>⌁</span><span>Predict Yield</span>";
    }
});

compareBtn.addEventListener("click", async () => {
    const crops = Array.from(document.getElementById("compareCrops").selectedOptions)
        .map(option => option.value);
    if (crops.length < 2 || crops.length > 8) {
        showError("Select between 2 and 8 crops to compare.");
        return;
    }

    const inputs = { ...readForm(), crops };
    const validationError = validateInput({ ...inputs, Crop: crops[0] });
    if (validationError) {
        showError(validationError);
        return;
    }

    compareBtn.disabled = true;
    compareBtn.textContent = "Comparing crops…";
    result.innerHTML = '<div class="result-info"><p>Applying the same conditions to each selected crop…</p></div>';
    try {
        const payload = await fetchJson("/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(inputs)
        }, 40000);

        result.replaceChildren();
        const heading = document.createElement("h3");
        heading.textContent = "Crop yield comparison";
        result.appendChild(heading);
        const caption = document.createElement("p");
        caption.className = "comparison-note";
        caption.textContent = `Ranked model estimates in ${payload.yield_unit || "dataset units"} using the same state, season, year and farm inputs. These are relative model outputs, not guaranteed harvest results.`;
        result.appendChild(caption);

        payload.predictions.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "comparison-row";
            const crop = document.createElement("span");
            crop.textContent = `${index + 1}. ${item.crop}`;
            const prediction = document.createElement("strong");
            prediction.textContent = `${Number(item.predicted_yield).toFixed(2)} ${item.yield_unit || payload.yield_unit || ""}`.trim();
            row.append(crop, prediction);
            result.appendChild(row);
        });

        const note = document.createElement("div");
        note.className = "result-info comparison-winner";
        const top = payload.predictions[0];
        note.textContent = `Highest model estimate in this comparison: ${top.crop}. Model estimates are not agronomic guarantees.`;
        result.appendChild(note);
    } catch (error) {
        showError(error.message);
    } finally {
        compareBtn.disabled = false;
        compareBtn.textContent = "Compare Selected Crops";
    }
});

document.getElementById("clearHistoryBtn").addEventListener("click", () => {
    try { localStorage.removeItem(HISTORY_KEY); } catch (_) {}
    renderHistory();
});

async function refreshApiStatus() {
    try {
        const health = await fetchJson("/health", {}, 12000);
        setApiStatus(health.status === "healthy", health.status === "healthy"
            ? "Prediction API online" : "Prediction API unavailable");
    } catch (_) {
        setApiStatus(false, "Prediction API unavailable");
    }
}

function showInitialResult() {
    result.innerHTML = `
        <div class="result-circle">
            <div class="plant-icon">🌱</div>
            <div class="yield-value">—</div>
            <div class="yield-unit">Predicted Yield</div>
        </div>
        <h3>Predicted Crop Yield</h3>
        <div class="result-info"><span>💡</span><p>Enter your farm parameters and predict a yield, or compare multiple crops under identical conditions.</p></div>
    `;
}

showInitialResult();
renderHistory();
loadOptions();
refreshApiStatus();

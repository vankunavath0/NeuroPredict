document.addEventListener("DOMContentLoaded", () => {
    const shell = document.querySelector(".mri-shell");
    const analyzeUrl = shell.dataset.analyzeUrl;

    const form = document.getElementById("mriAnalysisForm");
    const fileInput = document.getElementById("mriFile");
    const dropZone = document.getElementById("dropZone");
    const analyzeButton = document.getElementById("analyzeButton");
    const previewWrap = document.getElementById("previewWrap");
    const previewImage = document.getElementById("previewImage");
    const selectedFileName = document.getElementById("selectedFileName");

    const idleState = document.getElementById("idleState");
    const loadingState = document.getElementById("loadingState");
    const errorState = document.getElementById("errorState");
    const resultState = document.getElementById("resultState");

    const limeSamples = document.getElementById("limeSamples");
    const limeSamplesValue = document.getElementById("limeSamplesValue");
    const limeFeatures = document.getElementById("limeFeatures");
    const limeFeaturesValue = document.getElementById("limeFeaturesValue");
    const includeLime = document.getElementById("includeLime");

    let selectedFile = null;

    const setPanelState = (state) => {
        idleState.hidden = state !== "idle";
        loadingState.hidden = state !== "loading";
        errorState.hidden = state !== "error";
        resultState.hidden = state !== "result";
    };

    const showError = (message) => {
        errorState.textContent = message;
        setPanelState("error");
    };

    const syncRangeOutputs = () => {
        limeSamplesValue.textContent = limeSamples.value;
        limeFeaturesValue.textContent = limeFeatures.value;
    };

    const setSelectedFile = (file) => {
        if (!file) {
            selectedFile = null;
            analyzeButton.disabled = true;
            previewWrap.hidden = true;
            return;
        }

        if (!file.type.startsWith("image/")) {
            selectedFile = null;
            analyzeButton.disabled = true;
            previewWrap.hidden = true;
            showError("Please select a JPG or PNG MRI image.");
            return;
        }

        selectedFile = file;
        analyzeButton.disabled = false;
        selectedFileName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewWrap.hidden = false;
        };
        reader.readAsDataURL(file);
        setPanelState("idle");
    };

    fileInput.addEventListener("change", () => {
        setSelectedFile(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (!file) return;

        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        setSelectedFile(file);
    });

    limeSamples.addEventListener("input", syncRangeOutputs);
    limeFeatures.addEventListener("input", syncRangeOutputs);
    syncRangeOutputs();

    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-button").forEach((tab) => {
                tab.classList.toggle("active", tab === button);
            });
            document.querySelectorAll(".xai-image-wrap").forEach((panel) => {
                panel.hidden = panel.dataset.panel !== button.dataset.tab;
            });
        });
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = selectedFile || fileInput.files[0];
        if (!file) {
            showError("Select an MRI image before running analysis.");
            return;
        }

        const formData = new FormData();
        formData.append("mri_file", file);
        formData.append("lime_samples", limeSamples.value);
        formData.append("lime_features", limeFeatures.value);
        formData.append("include_lime", includeLime.checked ? "true" : "false");

        analyzeButton.disabled = true;
        setPanelState("loading");

        try {
            const response = await fetch(analyzeUrl, {
                method: "POST",
                body: formData,
                credentials: "same-origin",
            });
            const data = await response.json();

            if (!response.ok || data.status !== "success") {
                throw new Error(data.message || "MRI analysis failed.");
            }

            renderResult(data);
            setPanelState("result");
        } catch (error) {
            showError(error.message || "MRI analysis failed.");
        } finally {
            analyzeButton.disabled = false;
        }
    });

    const renderResult = (data) => {
        const prediction = data.prediction;
        const banner = document.getElementById("predictionBanner");
        const predictedClass = document.getElementById("predictedClass");
        const predictedDescription = document.getElementById("predictedDescription");
        const confidenceValue = document.getElementById("confidenceValue");
        const stageValue = document.getElementById("stageValue");
        const severityValue = document.getElementById("severityValue");
        const probabilitiesList = document.getElementById("probabilitiesList");
        const gradcamImage = document.getElementById("gradcamImage");
        const limeImage = document.getElementById("limeImage");
        const limeWarning = document.getElementById("limeWarning");

        banner.style.borderLeftColor = prediction.color;
        predictedClass.textContent = prediction.class_name;
        predictedClass.style.color = prediction.color;
        predictedDescription.textContent = prediction.description;
        confidenceValue.textContent = `${prediction.confidence_percent}%`;
        confidenceValue.style.color = prediction.color;
        stageValue.textContent = `${prediction.stage_index} / ${prediction.stage_total}`;
        severityValue.textContent = prediction.severity;
        severityValue.style.color = prediction.color;

        probabilitiesList.innerHTML = "";
        data.probabilities.forEach((item) => {
            const row = document.createElement("div");
            row.className = "probability-row";

            const label = document.createElement("span");
            label.textContent = item.class_name;

            const track = document.createElement("div");
            track.className = "probability-track";

            const fill = document.createElement("div");
            fill.className = "probability-fill";
            fill.style.width = `${Math.max(0, Math.min(item.percent, 100))}%`;
            fill.style.background = item.color;

            const percent = document.createElement("span");
            percent.className = "probability-percent";
            percent.textContent = `${item.percent}%`;
            percent.style.color = item.color;

            track.appendChild(fill);
            row.appendChild(label);
            row.appendChild(track);
            row.appendChild(percent);
            probabilitiesList.appendChild(row);
        });

        gradcamImage.src = data.images.gradcam;

        if (data.images.lime) {
            limeImage.hidden = false;
            limeImage.src = data.images.lime;
            limeWarning.hidden = Boolean(!data.lime_warning);
            limeWarning.textContent = data.lime_warning || "";
        } else {
            limeImage.hidden = true;
            limeWarning.hidden = false;
            limeWarning.textContent = data.lime_warning || "LIME generation was skipped for this run.";
        }

        document.querySelectorAll(".tab-button").forEach((button) => {
            const isGradcam = button.dataset.tab === "gradcam";
            button.classList.toggle("active", isGradcam);
        });
        document.querySelectorAll(".xai-image-wrap").forEach((panel) => {
            panel.hidden = panel.dataset.panel !== "gradcam";
        });
    };
});

const groundingPromptInput = document.getElementById("groundingPromptInput");
const groundingBoxThresholdInput = document.getElementById("groundingBoxThresholdInput");
const groundingTextThresholdInput = document.getElementById("groundingTextThresholdInput");
const groundingUseNmsInput = document.getElementById("groundingUseNmsInput");
const groundingNmsIouInput = document.getElementById("groundingNmsIouInput");

const imageListEl = document.getElementById("imageList");
const imageCountEl = document.getElementById("imageCount");
const currentImageNameEl = document.getElementById("currentImageName");
const statusTextEl = document.getElementById("statusText");
const refreshBtn = document.getElementById("refreshBtn");

const yoloDetectBtn = document.getElementById("yoloDetectBtn");
const yoloModelDropdownBtn = document.getElementById("yoloModelDropdownBtn");
const yoloModelDropdown = document.getElementById("yoloModelDropdown");
const groundingDinoDetectBtn = document.getElementById("groundingDinoDetectBtn");
let currentYoloModelId = null;
const imageSearchInput = document.getElementById("imageSearchInput");
const rejectImageBtn = document.getElementById("rejectImageBtn");

const canvas = document.getElementById("imageCanvas");
canvas.tabIndex = 0;
const ctx = canvas.getContext("2d");

const boxListEl = document.getElementById("boxList");
const boxCountEl = document.getElementById("boxCount");
const saveXmlBtn = document.getElementById("saveXmlBtn");
const deleteBoxBtn = document.getElementById("deleteBoxBtn");
const clearBoxesBtn = document.getElementById("clearBoxesBtn");
const backupAnnotationsBtn = document.getElementById("backupAnnotationsBtn");

const openPreviewWallBtn = document.getElementById("openPreviewWallBtn");
const backToLabelViewBtn = document.getElementById("backToLabelViewBtn");

const previewWallView = document.getElementById("previewWallView");
const previewWallGrid = document.getElementById("previewWallGrid");
const previewWallSearchInput = document.getElementById("previewWallSearchInput");
const previewWallRefreshBtn = document.getElementById("previewWallRefreshBtn");
const previewWallZoomSlider = document.getElementById("previewWallZoomSlider");
const previewWallZoomLabel = document.getElementById("previewWallZoomLabel");

const shortcutPreviousImageInput = document.getElementById("shortcutPreviousImageInput");
const shortcutNextImageInput = document.getElementById("shortcutNextImageInput");
const shortcutSaveXmlInput = document.getElementById("shortcutSaveXmlInput");
const shortcutSavePreviousImageInput = document.getElementById("shortcutSavePreviousImageInput");
const shortcutSaveNextImageInput = document.getElementById("shortcutSaveNextImageInput");
const shortcutYoloDetectInput = document.getElementById("shortcutYoloDetectInput");
const shortcutGroundingDinoDetectInput = document.getElementById("shortcutGroundingDinoDetectInput");
const shortcutDeleteBoxInput = document.getElementById("shortcutDeleteBoxInput");
const shortcutClearBoxesInput = document.getElementById("shortcutClearBoxesInput");
const shortcutReloadAnnotationInput = document.getElementById("shortcutReloadAnnotationInput");
const shortcutCancelSelectInput = document.getElementById("shortcutCancelSelectInput");

const saveShortcutConfigBtn = document.getElementById("saveShortcutConfigBtn");
const resetShortcutConfigBtn = document.getElementById("resetShortcutConfigBtn");
const shortcutRejectImageInput = document.getElementById("shortcutRejectImageInput");

let allImages = [];
let filteredImages = [];

let currentImageName = null;
let currentImage = null;

let boxes = [];
let boxInfos = [];
let selectedBoxIndex = -1;

let isDrawing = false;
let startPoint = null;
let currentPoint = null;

// ---- 深色模式 Canvas 颜色适配 ----
const canvasColors = {
    emptyHint: "#777777",
    selectedBox: "#ff0000",
    normalBox: "#00ff00",
    labelText: "#ff69b4",
    tempBox: "#0066ff",
    thumbnailBg: "#eeeeee",
    thumbnailErrorText: "#999999",
    previewBoxStroke: "#00ff00",
};

function updateCanvasColors() {
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        canvasColors.emptyHint = "#aaaaaa";
        canvasColors.selectedBox = "#ff4444";
        canvasColors.normalBox = "#44ff44";
        canvasColors.labelText = "#ff88cc";
        canvasColors.tempBox = "#4488ff";
        canvasColors.thumbnailBg = "#333333";
        canvasColors.thumbnailErrorText = "#888888";
        canvasColors.previewBoxStroke = "#44ff44";
    } else {
        canvasColors.emptyHint = "#777777";
        canvasColors.selectedBox = "#ff0000";
        canvasColors.normalBox = "#00ff00";
        canvasColors.labelText = "#ff69b4";
        canvasColors.tempBox = "#0066ff";
        canvasColors.thumbnailBg = "#eeeeee";
        canvasColors.thumbnailErrorText = "#999999";
        canvasColors.previewBoxStroke = "#00ff00";
    }
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    updateCanvasColors();
    redrawCanvas();
    renderImageList();
    if (typeof isPreviewWallVisible === 'function' && isPreviewWallVisible()) {
        renderPreviewWall();
    }
});

let previewWallZoom = 1.0;
let previewWallLoadedCount = 0;
let previewWallFiltered = [];
let previewWallSentinel = null;
let previewWallObserver = null;
let previewWallZoomTimer = null;
const PREVIEW_WALL_BATCH_SIZE = 24;

let imageView = {
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    scale: 1
};

const groundingDinoDefaultConfig = {
    boxThreshold: 0.18,
    textThreshold: 0.20,
    useNms: true,
    nmsIouThreshold: 0.20
};

const defaultShortcutConfig = {
    previousImage: "a",
    nextImage: "d",
    saveXml: "s",
    savePreviousImage: "q",
    saveNextImage: "e",
    yoloDetect: "y",
    groundingDinoDetect: "g",
    deleteBox: "delete",
    clearBoxes: "c",
    reloadAnnotation: "r",
    rejectImage:"x",
    cancelSelect: "0",
    selectBoxKeys: ["1","2","3","4","5","6","7","8","9"]
};

let shortcutConfig = loadShortcutConfig();


function initGroundingDinoInputs() {
    if (!groundingBoxThresholdInput) {
        return;
    }

    groundingBoxThresholdInput.value = String(groundingDinoDefaultConfig.boxThreshold);
    groundingTextThresholdInput.value = String(groundingDinoDefaultConfig.textThreshold);
    groundingUseNmsInput.checked = groundingDinoDefaultConfig.useNms;
    groundingNmsIouInput.value = String(groundingDinoDefaultConfig.nmsIouThreshold);
}


function loadShortcutConfig() {
    const text = localStorage.getItem("shortcutConfig");

    if (!text) {
        return {
            ...defaultShortcutConfig
        };
    }

    try {
        const userConfig = JSON.parse(text);

        return {
            ...defaultShortcutConfig,
            ...userConfig,
            selectBoxKeys: defaultShortcutConfig.selectBoxKeys
        };
    } catch (error) {
        console.error("快捷键配置读取失败：", error);

        return {
            ...defaultShortcutConfig
        };
    }
}


function saveShortcutConfig(config) {
    localStorage.setItem("shortcutConfig", JSON.stringify(config));
}


function normalizeShortcutKey(key) {
    const rawValue = String(key);

    if (rawValue === " ") {
        return "space";
    }

    let value = rawValue.trim().toLowerCase();

    if (value === "spacebar") {
        return "space";
    }

    if (value === "space") {
        return "space";
    }

    if (value === "del") {
        return "delete";
    }

    if (value === "esc") {
        return "escape";
    }

    if (value.startsWith("key") && value.length === 4) {
        return value.slice(3);
    }

    if (value.startsWith("digit") && value.length === 6) {
        return value.slice(5);
    }

    return value;
}


function shortcutMatches(shortcut, key) {
    if (Array.isArray(shortcut)) {
        return shortcut.some(item => normalizeShortcutKey(item) === key);
    }

    return normalizeShortcutKey(shortcut) === key;
}


function eventMatchesShortcut(event, shortcut) {
    const key = normalizeShortcutKey(event.key);
    const code = normalizeShortcutKey(event.code);

    return shortcutMatches(shortcut, key) || shortcutMatches(shortcut, code);
}


function getNumberInputValue(inputEl, defaultValue) {
    const text = String(inputEl.value).trim();

    if (text === "") {
        return defaultValue;
    }

    const value = Number(text);

    if (!Number.isFinite(value)) {
        return defaultValue;
    }

    return value;
}


function getShortcutInputBindings() {
    return [
        {
            key: "previousImage",
            input: shortcutPreviousImageInput,
            defaultValue: defaultShortcutConfig.previousImage
        },
        {
            key: "nextImage",
            input: shortcutNextImageInput,
            defaultValue: defaultShortcutConfig.nextImage
        },
        {
            key: "saveXml",
            input: shortcutSaveXmlInput,
            defaultValue: defaultShortcutConfig.saveXml
        },
        {
            key: "savePreviousImage",
            input: shortcutSavePreviousImageInput,
            defaultValue: defaultShortcutConfig.savePreviousImage
        },
        {
            key: "saveNextImage",
            input: shortcutSaveNextImageInput,
            defaultValue: defaultShortcutConfig.saveNextImage
        },
        {
            key: "yoloDetect",
            input: shortcutYoloDetectInput,
            defaultValue: defaultShortcutConfig.yoloDetect
        },
        {
            key: "groundingDinoDetect",
            input: shortcutGroundingDinoDetectInput,
            defaultValue: defaultShortcutConfig.groundingDinoDetect
        },
        {
            key: "deleteBox",
            input: shortcutDeleteBoxInput,
            defaultValue: defaultShortcutConfig.deleteBox
        },
        {
            key: "clearBoxes",
            input: shortcutClearBoxesInput,
            defaultValue: defaultShortcutConfig.clearBoxes
        },
        {
            key: "reloadAnnotation",
            input: shortcutReloadAnnotationInput,
            defaultValue: defaultShortcutConfig.reloadAnnotation
        },
        {
            key: "cancelSelect",
            input: shortcutCancelSelectInput,
            defaultValue: defaultShortcutConfig.cancelSelect
        },
        {
            key: "rejectImage",
            input: shortcutRejectImageInput,
            defaultValue: defaultShortcutConfig.rejectImage
        },
    ].filter(item => item.input);
}


function fillShortcutConfigInputs() {
    const bindings = getShortcutInputBindings();

    for (const item of bindings) {
        item.input.value = shortcutConfig[item.key];
    }
}


function bindShortcutInputCapture() {
    const bindings = getShortcutInputBindings();

    for (const item of bindings) {
        item.input.addEventListener("keydown", (event) => {
            if (event.key === "Tab") {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            const key = normalizeShortcutKey(event.key);

            item.input.value = key;
            statusTextEl.textContent = `状态：已设置 ${item.key} 快捷键为 ${key}，点击“保存快捷键”后生效`;
        });

        item.input.addEventListener("click", () => {
            item.input.select();
            statusTextEl.textContent = "状态：请按下想设置的快捷键";
        });
    }
}


function readShortcutConfigInputs() {
    const bindings = getShortcutInputBindings();

    const nextConfig = {
        ...shortcutConfig,
        selectBoxKeys: defaultShortcutConfig.selectBoxKeys
    };

    for (const item of bindings) {
        const value = normalizeShortcutKey(item.input.value.trim() || item.defaultValue);
        nextConfig[item.key] = value;
    }

    const conflictMessage = findShortcutConflictMessage(nextConfig);

    if (conflictMessage) {
        statusTextEl.textContent = conflictMessage;
        return;
    }

    shortcutConfig = nextConfig;
    saveShortcutConfig(shortcutConfig);
    fillShortcutConfigInputs();

    statusTextEl.textContent = "状态：快捷键配置已保存";
}


function resetShortcutConfig() {
    shortcutConfig = {
        ...defaultShortcutConfig
    };

    saveShortcutConfig(shortcutConfig);
    fillShortcutConfigInputs();

    statusTextEl.textContent = "状态：已恢复默认快捷键";
}


function findShortcutConflictMessage(config) {
    const keys = [
        config.previousImage,
        config.nextImage,
        config.saveXml,
        config.savePreviousImage,
        config.saveNextImage,
        config.yoloDetect,
        config.groundingDinoDetect,
        config.deleteBox,
        config.clearBoxes,
        config.reloadAnnotation,
        config.cancelSelect,
        config.rejectImage,
    ];

    const countMap = {};
    const duplicated = [];

    for (const key of keys) {
        const normalizedKey = normalizeShortcutKey(key);

        if (!countMap[normalizedKey]) {
            countMap[normalizedKey] = 0;
        }

        countMap[normalizedKey] += 1;
    }

    for (const key in countMap) {
        if (countMap[key] > 1) {
            duplicated.push(key);
        }
    }

    if (duplicated.length > 0) {
        return `状态：快捷键重复：${duplicated.join(", ")}，请修改后再保存`;
    }

    const reserved = defaultShortcutConfig.selectBoxKeys.map(key => normalizeShortcutKey(key));
    const usedReservedKeys = keys
        .map(key => normalizeShortcutKey(key))
        .filter(key => reserved.includes(key));

    if (usedReservedKeys.length > 0) {
        return `状态：${usedReservedKeys.join(", ")} 已固定用于选择框，不能设置为其他快捷键`;
    }

    return "";
}


function showPreviewWallView() {
    const mainLayout = document.querySelector(".main-layout");

    if (!previewWallView || !mainLayout) {
        return;
    }

    mainLayout.classList.add("hidden");
    previewWallView.classList.remove("hidden");

    if (previewWallZoomSlider) {
        previewWallZoomSlider.value = String(previewWallZoom);
    }
    updatePreviewWallZoom(previewWallZoom);
    renderPreviewWall();
}


function showLabelView() {
    const mainLayout = document.querySelector(".main-layout");

    if (!previewWallView || !mainLayout) {
        return;
    }

    if (previewWallObserver) {
        previewWallObserver.disconnect();
        previewWallObserver = null;
    }

    previewWallView.classList.add("hidden");
    mainLayout.classList.remove("hidden");

    resizeCanvasToDisplaySize();
    redrawCanvas();
}


function isPreviewWallVisible() {
    if (!previewWallView) {
        return false;
    }

    return !previewWallView.classList.contains("hidden");
}


async function loadImageList() {
    statusTextEl.textContent = "状态：正在加载图片列表...";

    try {
        const response = await fetch("/api/images");
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = "状态：加载图片列表失败";
            return;
        }

        allImages = data.images || [];
        renderImageList();

        if (isPreviewWallVisible()) {
            renderPreviewWall();
        }

        statusTextEl.textContent = "状态：图片列表加载完成";
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：无法连接后端";
    }
}


async function loadYoloModels() {
    try {
        const response = await fetch("/api/yolo_models");
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = "状态：加载 YOLO 模型列表失败";
            return;
        }

        yoloModelDropdown.innerHTML = "";

        if (data.default_model !== undefined && data.default_model !== null) {
            currentYoloModelId = String(data.default_model);
        } else if (data.models.length > 0) {
            currentYoloModelId = String(data.models[0].id);
        }

        for (const model of data.models) {
            const li = document.createElement("li");
            li.textContent = model.name;
            li.dataset.modelId = model.id;

            if (String(model.id) === currentYoloModelId) {
                li.classList.add("selected");
                yoloDetectBtn.textContent = model.name;
            }

            li.addEventListener("click", (event) => {
                event.stopPropagation();
                currentYoloModelId = String(model.id);
                yoloDetectBtn.textContent = model.name;

                yoloModelDropdown.querySelectorAll("li").forEach(item => {
                    item.classList.toggle("selected", item.dataset.modelId === currentYoloModelId);
                });

                closeYoloModelDropdown();
            });

            yoloModelDropdown.appendChild(li);
        }
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：无法加载 YOLO 模型列表";
    }
}

function toggleYoloModelDropdown() {
    if (!yoloModelDropdown) {
        return;
    }

    yoloModelDropdown.classList.toggle("hidden");
}

function closeYoloModelDropdown() {
    if (!yoloModelDropdown) {
        return;
    }

    yoloModelDropdown.classList.add("hidden");
}


function renderImageList() {
    imageListEl.innerHTML = "";

    const keyword = imageSearchInput.value.trim().toLowerCase();

    filteredImages = allImages.filter(imageName => {
        if (!keyword) {
            return true;
        }

        return imageName.toLowerCase().includes(keyword);
    });

    imageCountEl.textContent = `图片数量：${filteredImages.length} / ${allImages.length}`;

    if (filteredImages.length === 0) {
        const li = document.createElement("li");
        li.textContent = "没有匹配的图片";
        imageListEl.appendChild(li);
        return;
    }

    for (const imageName of filteredImages) {
        const li = document.createElement("li");
        li.classList.add("image-item");
        li.dataset.imageName = imageName;

        if (imageName === currentImageName) {
            li.classList.add("active");
        }

        const previewCanvas = document.createElement("canvas");
        previewCanvas.classList.add("thumbnail-canvas");
        previewCanvas.width = 96;
        previewCanvas.height = 72;

        const nameDiv = document.createElement("div");
        nameDiv.classList.add("image-item-name");
        nameDiv.textContent = imageName;

        li.appendChild(previewCanvas);
        li.appendChild(nameDiv);

        li.addEventListener("click", () => {
            selectImage(imageName);
        });

        requestAnimationFrame(() => {
            scrollCurrentImageIntoView();
        });

        imageListEl.appendChild(li);

        drawThumbnail(imageName, previewCanvas);
    }
}


function getPreviewCanvasDims(zoom) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    return {
        w: Math.round(120 * zoom * dpr),
        h: Math.round(80 * zoom * dpr)
    };
}

function updatePreviewWallZoom(value) {
    previewWallZoom = value;
    previewWallGrid.style.setProperty("--zoom", String(value));
    if (previewWallZoomLabel) {
        previewWallZoomLabel.textContent = Math.round(value * 100) + "%";
    }
}

function createPreviewWallItem(imageName) {
    const item = document.createElement("div");
    item.classList.add("preview-wall-item");
    item.dataset.imageName = imageName;

    if (imageName === currentImageName) {
        item.classList.add("active");
    }

    const dims = getPreviewCanvasDims(previewWallZoom);
    const wallCanvas = document.createElement("canvas");
    wallCanvas.width = dims.w;
    wallCanvas.height = dims.h;

    const nameDiv = document.createElement("div");
    nameDiv.classList.add("preview-wall-name");
    nameDiv.textContent = imageName;

    item.appendChild(wallCanvas);
    item.appendChild(nameDiv);

    item.addEventListener("click", () => {
        showLabelView();
        selectImage(imageName);
    });

    return item;
}

function createPreviewWallSentinel() {
    const sentinel = document.createElement("div");
    sentinel.classList.add("preview-wall-sentinel");
    return sentinel;
}

function loadNextPreviewWallBatch() {
    if (previewWallLoadedCount >= previewWallFiltered.length) {
        if (previewWallObserver && previewWallSentinel) {
            previewWallObserver.unobserve(previewWallSentinel);
        }
        return;
    }

    const startIndex = previewWallLoadedCount;
    const endIndex = Math.min(startIndex + PREVIEW_WALL_BATCH_SIZE, previewWallFiltered.length);
    const batch = previewWallFiltered.slice(startIndex, endIndex);

    if (batch.length === 0) {
        return;
    }

    if (previewWallObserver && previewWallSentinel) {
        previewWallObserver.unobserve(previewWallSentinel);
    }
    if (previewWallSentinel && previewWallSentinel.parentNode) {
        previewWallSentinel.remove();
    }
    previewWallSentinel = null;

    for (const imageName of batch) {
        const item = createPreviewWallItem(imageName);
        previewWallGrid.appendChild(item);
        const canvas = item.querySelector("canvas");
        drawThumbnail(imageName, canvas);
    }

    previewWallLoadedCount = endIndex;

    previewWallSentinel = createPreviewWallSentinel();
    previewWallGrid.appendChild(previewWallSentinel);

    if (previewWallObserver) {
        previewWallObserver.observe(previewWallSentinel);
    }
}

function setupPreviewWallObserver() {
    if (previewWallObserver) {
        previewWallObserver.disconnect();
    }

    previewWallObserver = new IntersectionObserver((entries) => {
        if (entries.some(e => e.isIntersecting)) {
            loadNextPreviewWallBatch();
        }
    }, {
        root: previewWallGrid,
        rootMargin: "300px",
        threshold: 0
    });

    if (previewWallSentinel) {
        previewWallObserver.observe(previewWallSentinel);
    }
}

function redrawAllPreviewCanvases() {
    const dims = getPreviewCanvasDims(previewWallZoom);
    const items = previewWallGrid.querySelectorAll(".preview-wall-item");

    for (const item of items) {
        const canvas = item.querySelector("canvas");
        if (!canvas) {
            continue;
        }

        canvas.width = dims.w;
        canvas.height = dims.h;

        const imageName = item.dataset.imageName;
        if (imageName) {
            drawThumbnail(imageName, canvas);
        }
    }
}

function renderPreviewWall() {
    if (!previewWallGrid) {
        return;
    }

    if (previewWallObserver) {
        previewWallObserver.disconnect();
        previewWallObserver = null;
    }

    previewWallGrid.innerHTML = "";

    const keyword = previewWallSearchInput ? previewWallSearchInput.value.trim().toLowerCase() : "";

    previewWallFiltered = allImages.filter(imageName => {
        if (!keyword) {
            return true;
        }
        return imageName.toLowerCase().includes(keyword);
    });

    previewWallLoadedCount = 0;
    previewWallSentinel = null;

    if (previewWallFiltered.length === 0) {
        const emptyDiv = document.createElement("div");
        emptyDiv.classList.add("preview-wall-empty");
        emptyDiv.textContent = "没有匹配的图片";
        previewWallGrid.appendChild(emptyDiv);
        return;
    }

    previewWallSentinel = createPreviewWallSentinel();
    previewWallGrid.appendChild(previewWallSentinel);

    setupPreviewWallObserver();
    loadNextPreviewWallBatch();
}


async function drawThumbnail(imageName, previewCanvas) {
    const previewCtx = previewCanvas.getContext("2d");

    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.fillStyle = canvasColors.thumbnailBg;
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);

    const img = new Image();

    img.onload = async () => {
        const imgW = img.naturalWidth;
        const imgH = img.naturalHeight;

        const scale = Math.min(
            previewCanvas.width / imgW,
            previewCanvas.height / imgH
        );

        const drawW = imgW * scale;
        const drawH = imgH * scale;
        const offsetX = (previewCanvas.width - drawW) / 2;
        const offsetY = (previewCanvas.height - drawH) / 2;

        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

        previewCtx.drawImage(
            img,
            offsetX,
            offsetY,
            drawW,
            drawH
        );

        try {
            const response = await fetch(`/api/annotation/${encodeURIComponent(imageName)}`);
            const data = await response.json();

            if (!data.success) {
                return;
            }

            const result = data.result;
            const previewBoxes = result.boxes || [];

            previewCtx.lineWidth = 1;
            previewCtx.strokeStyle = canvasColors.previewBoxStroke;

            for (const box of previewBoxes) {
                const x1 = offsetX + box[0] * scale;
                const y1 = offsetY + box[1] * scale;
                const x2 = offsetX + box[2] * scale;
                const y2 = offsetY + box[3] * scale;

                previewCtx.strokeRect(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                );
            }
        } catch (error) {
            console.error("缩略图标注读取失败：", imageName, error);
        }
    };

    img.onerror = () => {
        previewCtx.fillStyle = canvasColors.thumbnailErrorText;
        previewCtx.font = "12px Arial";
        previewCtx.textAlign = "center";
        previewCtx.textBaseline = "middle";
        previewCtx.fillText(
            "加载失败",
            previewCanvas.width / 2,
            previewCanvas.height / 2
        );
    };

    img.src = `/api/image/${encodeURIComponent(imageName)}`;
}


function getImageItemLi(imageName) {
    const items = document.querySelectorAll("#imageList li");

    for (const li of items) {
        if (li.dataset.imageName === imageName) {
            return li;
        }
    }

    return null;
}

function scrollCurrentImageIntoView() {
    if (!currentImageName) {
        return;
    }

    const activeLi = getImageItemLi(currentImageName);

    if (!activeLi) {
        return;
    }

    activeLi.scrollIntoView({
        block: "nearest",
        inline: "nearest",
        behavior: "smooth"
    });
}

function refreshThumbnail(imageName) {
    const li = getImageItemLi(imageName);

    if (!li) {
        return;
    }

    const previewCanvas = li.querySelector("canvas");

    if (!previewCanvas) {
        return;
    }

    drawThumbnail(imageName, previewCanvas);
}


function selectImage(imageName) {
    currentImageName = imageName;
    boxes = [];
    boxInfos = [];
    selectedBoxIndex = -1;

    document.querySelectorAll("#imageList li").forEach(li => {
        li.classList.remove("active");
    });

    const activeLi = getImageItemLi(imageName);

    if (activeLi) {
        activeLi.classList.add("active");

        requestAnimationFrame(() => {
            scrollCurrentImageIntoView();
        });
    }

    currentImage = new Image();

    currentImage.onload = async () => {
        resizeCanvasToDisplaySize();
        redrawCanvas();
        renderBoxList();

        currentImageNameEl.textContent = `当前图片：${imageName}`;
        statusTextEl.textContent = `状态：图片已加载，原图尺寸 ${currentImage.naturalWidth} × ${currentImage.naturalHeight}`;

        await loadAnnotation(imageName);
    };

    currentImage.onerror = () => {
        statusTextEl.textContent = "状态：图片加载失败";
    };

    currentImage.src = `/api/image/${encodeURIComponent(imageName)}`;
}


function selectPreviousImage() {
    if (filteredImages.length === 0) {
        return;
    }

    let index = filteredImages.indexOf(currentImageName);

    if (index <= 0) {
        index = filteredImages.length;
    }

    selectImage(filteredImages[index - 1]);
}


function selectNextImage() {
    if (filteredImages.length === 0) {
        return;
    }

    let index = filteredImages.indexOf(currentImageName);

    if (index === -1 || index >= filteredImages.length - 1) {
        index = -1;
    }

    selectImage(filteredImages[index + 1]);
}


function resizeCanvasToDisplaySize() {
    const rect = canvas.getBoundingClientRect();

    canvas.width = Math.floor(rect.width);
    canvas.height = Math.floor(rect.height);
}


function updateImageView() {
    if (!currentImage) {
        return;
    }

    const imgW = currentImage.naturalWidth;
    const imgH = currentImage.naturalHeight;

    const canvasW = canvas.width;
    const canvasH = canvas.height;

    const scale = Math.min(canvasW / imgW, canvasH / imgH);

    const drawWidth = imgW * scale;
    const drawHeight = imgH * scale;

    const offsetX = (canvasW - drawWidth) / 2;
    const offsetY = (canvasH - drawHeight) / 2;

    imageView = {
        offsetX,
        offsetY,
        drawWidth,
        drawHeight,
        scale
    };
}


function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!currentImage) {
        drawEmptyHint();
        return;
    }

    updateImageView();

    ctx.drawImage(
        currentImage,
        imageView.offsetX,
        imageView.offsetY,
        imageView.drawWidth,
        imageView.drawHeight
    );

    drawAllBoxes();

    if (isDrawing && startPoint && currentPoint) {
        drawTempBox();
    }
}


function drawEmptyHint() {
    ctx.fillStyle = canvasColors.emptyHint;
    ctx.font = "18px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("请从左侧选择一张图片", canvas.width / 2, canvas.height / 2);
}


function drawAllBoxes() {
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const canvasBox = imageBoxToCanvasBox(box);

        const isSelected = i === selectedBoxIndex;

        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeStyle = isSelected ? canvasColors.selectedBox : canvasColors.normalBox;

        ctx.strokeRect(
            canvasBox.x,
            canvasBox.y,
            canvasBox.w,
            canvasBox.h
        );

        const labelText = ` ${i + 1}`;

        ctx.font = "bold 14px Arial";

        const labelPaddingX = 0;
        const labelHeight = 20;

        const labelX = canvasBox.x;
        const labelY = Math.max(canvasBox.y - labelHeight, 0);

        ctx.fillStyle = canvasColors.labelText;
        ctx.fillText(
            labelText,
            labelX + labelPaddingX,
            labelY + 15
        );
    }
}


function drawTempBox() {
    const x = Math.min(startPoint.x, currentPoint.x);
    const y = Math.min(startPoint.y, currentPoint.y);
    const w = Math.abs(currentPoint.x - startPoint.x);
    const h = Math.abs(currentPoint.y - startPoint.y);

    ctx.lineWidth = 2;
    ctx.strokeStyle = canvasColors.tempBox;
    ctx.strokeRect(x, y, w, h);
}


function getMousePosition(event) {
    const rect = canvas.getBoundingClientRect();

    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}


function isPointInsideDisplayedImage(point) {
    return (
        point.x >= imageView.offsetX &&
        point.x <= imageView.offsetX + imageView.drawWidth &&
        point.y >= imageView.offsetY &&
        point.y <= imageView.offsetY + imageView.drawHeight
    );
}


function canvasPointToImagePoint(point) {
    const x = (point.x - imageView.offsetX) / imageView.scale;
    const y = (point.y - imageView.offsetY) / imageView.scale;

    return {
        x: clamp(Math.round(x), 0, currentImage.naturalWidth),
        y: clamp(Math.round(y), 0, currentImage.naturalHeight)
    };
}


function imagePointToCanvasPoint(point) {
    return {
        x: imageView.offsetX + point.x * imageView.scale,
        y: imageView.offsetY + point.y * imageView.scale
    };
}


function imageBoxToCanvasBox(box) {
    const p1 = imagePointToCanvasPoint({ x: box[0], y: box[1] });
    const p2 = imagePointToCanvasPoint({ x: box[2], y: box[3] });

    return {
        x: p1.x,
        y: p1.y,
        w: p2.x - p1.x,
        h: p2.y - p1.y
    };
}


function clamp(value, minValue, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
}


function makeImageBoxFromCanvasPoints(p1, p2) {
    const imgP1 = canvasPointToImagePoint(p1);
    const imgP2 = canvasPointToImagePoint(p2);

    const xmin = Math.min(imgP1.x, imgP2.x);
    const ymin = Math.min(imgP1.y, imgP2.y);
    const xmax = Math.max(imgP1.x, imgP2.x);
    const ymax = Math.max(imgP1.y, imgP2.y);

    return [xmin, ymin, xmax, ymax];
}

function makeDefaultBoxInfo() {
    return {
        name: "object",
        score: null
    };
}


function normalizeObjectsToBoxesAndInfos(objects) {
    const nextBoxes = [];
    const nextBoxInfos = [];

    for (const obj of objects) {
        if (!obj.box || obj.box.length !== 4) {
            continue;
        }

        nextBoxes.push(obj.box);

        nextBoxInfos.push({
            name: obj.name || "object",
            score: typeof obj.score === "number" ? obj.score : null
        });
    }

    return {
        boxes: nextBoxes,
        boxInfos: nextBoxInfos
    };
}

function renderBoxList() {
    boxListEl.innerHTML = "";
    boxCountEl.textContent = `框数量：${boxes.length}`;

    if (boxes.length === 0) {
        const li = document.createElement("li");
        li.textContent = "暂无框，请在图片上拖动画框";
        boxListEl.appendChild(li);
        return;
    }

    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const info = boxInfos[i] || makeDefaultBoxInfo();

        const li = document.createElement("li");
        li.classList.add("box-list-item");

        if (i === selectedBoxIndex) {
            li.classList.add("active");
        }

        const scoreText = info.score === null || info.score === undefined
            ? "-"
            : Number(info.score).toFixed(3);

        li.innerHTML = `
            <div class="box-list-title">
                <span>${i + 1}. ${escapeHtml(info.name)}</span>
                <span>score: ${scoreText}</span>
            </div>
            <div class="box-list-coord">
                [${box[0]}, ${box[1]}, ${box[2]}, ${box[3]}]
            </div>
        `;

        li.addEventListener("click", () => {
            selectBoxByIndex(i);
        });

        boxListEl.appendChild(li);
    }
}


function escapeHtml(text) {
    return String(text)
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;")
        .replaceAll('"',"&quot;")
        .replaceAll("'","&#039;");
}


function selectBoxByIndex(index) {
    if (index < 0 || index >= boxes.length) {
        statusTextEl.textContent = `状态：第 ${index + 1} 个框不存在`;
        return;
    }

    selectedBoxIndex = index;

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = `状态：已选中第 ${index + 1} 个框`;
}


function cancelSelectedBox() {
    selectedBoxIndex = -1;

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = "状态：已取消选中框";
}


async function saveXml() {
    if (!currentImageName) {
        statusTextEl.textContent = "状态：请先选择一张图片";
        return false;
    }

    const payload = {
        image_name: currentImageName,
        boxes: boxes
    };

    statusTextEl.textContent = "状态：正在保存 XML...";

    try {
        const response = await fetch("/api/save", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `状态：保存失败，${data.message}`;
            return false;
        }

        const result = data.result;
        statusTextEl.textContent = `状态：保存成功，${result.xml_name}，框数量 ${result.box_count}`;

        refreshThumbnail(currentImageName);

        return true;
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：保存失败，无法连接后端";
        return false;
    }
}

async function rejectCurrentImage() {
    if (!currentImageName) {
        statusTextEl.textContent = "状态：请先选择一张图片";
        return;
    }

    const imageNameToReject = currentImageName;
    const currentIndex = filteredImages.indexOf(imageNameToReject);

    const confirmed = window.confirm(
        `确认把 ${imageNameToReject} 移入废弃区吗？\n对应 XML 也会一起移动。`
    );

    if (!confirmed) {
        return;
    }

    statusTextEl.textContent = `状态：正在移入废弃区：${imageNameToReject}`;

    try {
        const response = await fetch("/api/reject_image", {
            method: "POST",
            headers: {
                "Content-Type":"application/json"
            },
            body: JSON.stringify({
                image_name:imageNameToReject
            })
        });

        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `状态：移入废弃区失败，${data.message}`;
            return;
        }

        allImages = allImages.filter(name => name !== imageNameToReject);
        filteredImages = filteredImages.filter(name => name !== imageNameToReject);

        boxes = [];
        boxInfos = [];
        selectedBoxIndex = -1;
        currentImage = null;
        currentImageName = null;

        renderImageList();
        renderBoxList();
        redrawCanvas();

        statusTextEl.textContent = `状态：已移入废弃区：${imageNameToReject}`;

        if (filteredImages.length > 0) {
            let nextIndex = currentIndex;

            if (nextIndex >= filteredImages.length) {
                nextIndex = filteredImages.length - 1;
            }

            selectImage(filteredImages[nextIndex]);
        } else {
            currentImageNameEl.textContent = "当前图片：无";
        }

        if (typeof renderPreviewWall === "function" && isPreviewWallVisible()) {
            renderPreviewWall();
        }
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：移入废弃区失败，无法连接后端";
    }
}

async function saveAndSelectPreviousImage() {
    const success = await saveXml();

    if (!success) {
        return;
    }

    selectPreviousImage();
}


async function saveAndSelectNextImage() {
    const success = await saveXml();

    if (!success) {
        return;
    }

    selectNextImage();
}


async function backupAnnotations() {
    statusTextEl.textContent = "状态：正在备份 Annotations...";

    try {
        const response = await fetch("/api/backup_annotations");

        if (!response.ok) {
            let message = "备份失败";

            try {
                const data = await response.json();
                message = data.message || message;
            } catch {
                // 忽略 JSON 解析失败
            }

            statusTextEl.textContent = `状态：${message}`;
            return;
        }

        const blob = await response.blob();

        let fileName = "Annotations_backup.zip";

        const disposition = response.headers.get("Content-Disposition");

        if (disposition) {
            const match = disposition.match(/filename="?([^"]+)"?/);

            if (match && match[1]) {
                fileName = decodeURIComponent(match[1]);
            }
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");

        a.href = url;
        a.download = fileName;

        document.body.appendChild(a);
        a.click();
        a.remove();

        window.URL.revokeObjectURL(url);

        statusTextEl.textContent = `状态：备份完成，${fileName}`;
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：备份失败，无法连接后端";
    }
}


async function loadAnnotation(imageName) {
    try {
        const response = await fetch(`/api/annotation/${encodeURIComponent(imageName)}`);
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `状态：读取 XML 失败，${data.message}`;
            return;
        }

        const result = data.result;

        boxes = result.boxes || [];
        boxInfos = boxes.map(() => makeDefaultBoxInfo());
        selectedBoxIndex = -1;

        renderBoxList();
        redrawCanvas();

        if (result.exists) {
            statusTextEl.textContent = `状态：已读取 ${result.xml_name}，框数量 ${boxes.length}`;
        } else {
            statusTextEl.textContent = "状态：当前图片暂无 XML，可新建标注";
        }
    } catch (error) {
        console.error("读取 XML 失败：", error);
        statusTextEl.textContent = "状态：读取 XML 失败，无法连接后端";
    }
}


async function autoDetect(detector, yoloModelId) {
    if (!currentImageName) {
        statusTextEl.textContent = "状态：请先选择一张图片";
        return;
    }

    const payload = {
        image_name: currentImageName,
        detector: detector
    };

    if (detector === "yolo") {
        payload.model_select = yoloModelId || currentYoloModelId;
    }

    if (detector === "grounding_dino") {
        payload.prompt = groundingPromptInput.value.trim();

        payload.box_threshold = getNumberInputValue(
            groundingBoxThresholdInput,
            groundingDinoDefaultConfig.boxThreshold
        );

        payload.text_threshold = getNumberInputValue(
            groundingTextThresholdInput,
            groundingDinoDefaultConfig.textThreshold
        );

        payload.use_nms = groundingUseNmsInput.checked;

        payload.nms_iou_threshold = getNumberInputValue(
            groundingNmsIouInput,
            groundingDinoDefaultConfig.nmsIouThreshold
        );
    }

    const detectorName = detector === "yolo"
        ? `YOLO模型 ${payload.model_select}`
        : "Grounding DINO";

    statusTextEl.textContent = `状态：正在使用 ${detectorName} 自动标注...`;

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        console.log("自动标注返回：", data);

        if (!data.success) {
            statusTextEl.textContent = `状态：自动标注失败，${data.message}`;
            return;
        }

        if (Array.isArray(data.objects) && data.objects.length > 0) {
            const normalized = normalizeObjectsToBoxesAndInfos(data.objects);

            boxes = normalized.boxes;
            boxInfos = normalized.boxInfos;
        } else {
            boxes = data.boxes || [];
            boxInfos = boxes.map(() => makeDefaultBoxInfo());
        }

        selectedBoxIndex = -1;

        renderBoxList();
        redrawCanvas();
        canvas.focus();

        statusTextEl.textContent = `状态：${detectorName} 标注完成，框数量 ${boxes.length}`;
    } catch (error) {
        console.error("自动标注请求失败：", error);
        statusTextEl.textContent = "状态：自动标注失败，无法连接后端";
    }
}


function deleteSelectedBox() {
    if (boxes.length === 0) {
        statusTextEl.textContent = "状态：当前没有框可删除";
        return;
    }

    if (selectedBoxIndex < 0 || selectedBoxIndex >= boxes.length) {
        selectedBoxIndex = boxes.length - 1;
    }

    const deletedIndex = selectedBoxIndex;

    boxes.splice(selectedBoxIndex, 1);
    boxInfos.splice(selectedBoxIndex, 1);

    if (boxes.length === 0) {
        selectedBoxIndex = -1;
    } else if (deletedIndex >= boxes.length) {
        selectedBoxIndex = boxes.length - 1;
    } else {
        selectedBoxIndex = deletedIndex;
    }

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = `状态：已删除第 ${deletedIndex + 1} 个框`;
}


function clearAllBoxes() {
    boxes = [];
    boxInfos = [];
    selectedBoxIndex = -1;

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = "状态：已清空所有框";
}


canvas.addEventListener("mousedown", (event) => {
    canvas.focus();

    if (!currentImage) {
        return;
    }

    resizeCanvasToDisplaySize();
    updateImageView();

    const point = getMousePosition(event);

    if (!isPointInsideDisplayedImage(point)) {
        return;
    }

    isDrawing = true;
    startPoint = point;
    currentPoint = point;
    selectedBoxIndex = -1;

    redrawCanvas();
});


canvas.addEventListener("mousemove", (event) => {
    if (!isDrawing) {
        return;
    }

    currentPoint = getMousePosition(event);

    currentPoint.x = clamp(
        currentPoint.x,
        imageView.offsetX,
        imageView.offsetX + imageView.drawWidth
    );

    currentPoint.y = clamp(
        currentPoint.y,
        imageView.offsetY,
        imageView.offsetY + imageView.drawHeight
    );

    redrawCanvas();
});


canvas.addEventListener("mouseup", () => {
    if (!isDrawing) {
        return;
    }

    isDrawing = false;

    const newBox = makeImageBoxFromCanvasPoints(startPoint, currentPoint);

    const width = newBox[2] - newBox[0];
    const height = newBox[3] - newBox[1];

    if (width >= 3 && height >= 3) {
        boxes.push(newBox);
        boxInfos.push(makeDefaultBoxInfo());
        selectedBoxIndex = boxes.length - 1;
    }

    startPoint = null;
    currentPoint = null;

    renderBoxList();
    redrawCanvas();
});


canvas.addEventListener("mouseleave", () => {
    if (isDrawing) {
        isDrawing = false;
        startPoint = null;
        currentPoint = null;
        redrawCanvas();
    }
});


deleteBoxBtn.addEventListener("click", deleteSelectedBox);
clearBoxesBtn.addEventListener("click", clearAllBoxes);

if (backupAnnotationsBtn) {
    backupAnnotationsBtn.addEventListener("click", backupAnnotations);
}

if (openPreviewWallBtn) {
    openPreviewWallBtn.addEventListener("click", showPreviewWallView);
}

if (backToLabelViewBtn) {
    backToLabelViewBtn.addEventListener("click", showLabelView);
}

if (saveShortcutConfigBtn) {
    saveShortcutConfigBtn.addEventListener("click", readShortcutConfigInputs);
}

if (resetShortcutConfigBtn) {
    resetShortcutConfigBtn.addEventListener("click", resetShortcutConfig);
}

if (rejectImageBtn) {
    rejectImageBtn.addEventListener("click", rejectCurrentImage);
}

yoloDetectBtn.addEventListener("click", () => {
    autoDetect("yolo", currentYoloModelId);
});

if (yoloModelDropdownBtn) {
    yoloModelDropdownBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleYoloModelDropdown();
    });
}

document.addEventListener("click", (event) => {
    if (!yoloModelDropdown || !yoloModelDropdownBtn) {
        return;
    }

    if (!yoloModelDropdownBtn.contains(event.target) && !yoloModelDropdown.contains(event.target)) {
        closeYoloModelDropdown();
    }
});

groundingDinoDetectBtn.addEventListener("click", () => {
    autoDetect("grounding_dino");
});

refreshBtn.addEventListener("click", loadImageList);
saveXmlBtn.addEventListener("click", saveXml);

imageSearchInput.addEventListener("input", () => {
    renderImageList();
});

if (previewWallSearchInput) {
    previewWallSearchInput.addEventListener("input", () => {
        renderPreviewWall();
    });
}

if (previewWallRefreshBtn) {
    previewWallRefreshBtn.addEventListener("click", () => {
        renderPreviewWall();
    });
}

if (previewWallZoomSlider) {
    previewWallZoomSlider.addEventListener("input", () => {
        const value = parseFloat(previewWallZoomSlider.value);
        updatePreviewWallZoom(value);

        if (previewWallZoomTimer) {
            clearTimeout(previewWallZoomTimer);
        }
        previewWallZoomTimer = setTimeout(() => {
            redrawAllPreviewCanvases();
        }, 250);
    });
}

if (previewWallGrid) {
    previewWallGrid.addEventListener("wheel", (event) => {
        if (!event.ctrlKey) {
            return;
        }
        event.preventDefault();

        const delta = event.deltaY > 0 ? -0.1 : 0.1;
        let newZoom = previewWallZoom + delta;
        newZoom = Math.max(0.4, Math.min(2.5, newZoom));
        newZoom = Math.round(newZoom * 10) / 10;

        if (previewWallZoomSlider) {
            previewWallZoomSlider.value = String(newZoom);
            previewWallZoomSlider.dispatchEvent(new Event("input"));
        }
    }, { passive: false });
}


document.addEventListener("keydown", async (event) => {
    const target = event.target;
    const tagName = target.tagName.toLowerCase();

    if (tagName === "input" || tagName === "select" || tagName === "textarea") {
        return;
    }

    const selectBoxIndex = shortcutConfig.selectBoxKeys.findIndex(shortcut => {
        return eventMatchesShortcut(event, shortcut);
    });

    if (selectBoxIndex !== -1) {
        event.preventDefault();
        selectBoxByIndex(selectBoxIndex);
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.cancelSelect)) {
        event.preventDefault();
        cancelSelectedBox();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.savePreviousImage)) {
        event.preventDefault();
        await saveAndSelectPreviousImage();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.saveNextImage)) {
        event.preventDefault();
        await saveAndSelectNextImage();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.saveXml)) {
        event.preventDefault();
        await saveXml();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.previousImage)) {
        event.preventDefault();
        selectPreviousImage();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.nextImage)) {
        event.preventDefault();
        selectNextImage();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.yoloDetect)) {
        event.preventDefault();
        autoDetect("yolo", currentYoloModelId);
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.groundingDinoDetect)) {
        event.preventDefault();
        autoDetect("grounding_dino");
        return;
    }

    if (
        eventMatchesShortcut(event, shortcutConfig.deleteBox) ||
        eventMatchesShortcut(event, "backspace")
    ) {
        event.preventDefault();
        deleteSelectedBox();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.clearBoxes)) {
        event.preventDefault();
        clearAllBoxes();
        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.reloadAnnotation)) {
        event.preventDefault();

        if (currentImageName) {
            loadAnnotation(currentImageName);
        }

        return;
    }

    if (eventMatchesShortcut(event, shortcutConfig.rejectImage)) {
        event.preventDefault();
        await rejectCurrentImage();
        return;
    }
});


window.addEventListener("resize", () => {
    resizeCanvasToDisplaySize();
    redrawCanvas();
});


// 页面打开后初始化
updateCanvasColors();
resizeCanvasToDisplaySize();
redrawCanvas();
initGroundingDinoInputs();
fillShortcutConfigInputs();
bindShortcutInputCapture();
loadYoloModels();
loadImageList();
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
const filterDefaultClassBtn = document.getElementById("filterDefaultClassBtn");

const yoloDetectBtn = document.getElementById("yoloDetectBtn");
const yoloModelDropdownBtn = document.getElementById("yoloModelDropdownBtn");
const yoloModelDropdown = document.getElementById("yoloModelDropdown");
const groundingDinoDetectBtn = document.getElementById("groundingDinoDetectBtn");
const pureColorClassifyBtn = document.getElementById("pureColorClassifyBtn");
const modelClassifyBtn = document.getElementById("modelClassifyBtn");
const classifyModelDropdownBtn = document.getElementById("classifyModelDropdownBtn");
const classifyModelDropdown = document.getElementById("classifyModelDropdown");
const pickColorBtn = document.getElementById("pickColorBtn");
let currentYoloModelId = null;
let yoloModelCount = 0;
let currentClassifyModelId = null;
let classifyModelCount = 0;
const imageSearchInput = document.getElementById("imageSearchInput");
const rejectImageBtn = document.getElementById("rejectImageBtn");

const canvas = document.getElementById("imageCanvas");
canvas.tabIndex = 0;
const ctx = canvas.getContext("2d");
const canvasClassMenu = document.getElementById("canvasClassMenu");

const boxListEl = document.getElementById("boxList");
const boxCountEl = document.getElementById("boxCount");
const saveXmlBtn = document.getElementById("saveXmlBtn");
const deleteBoxBtn = document.getElementById("deleteBoxBtn");
const clearBoxesBtn = document.getElementById("clearBoxesBtn");
const backupAnnotationsBtn = document.getElementById("backupAnnotationsBtn");

const jumpToImageInput = document.getElementById("jumpToImageInput");
const positionProgressEl = document.getElementById("positionProgress");

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
const shortcutYoloDetectInput = document.getElementById("shortcutYoloDetectInput");
const shortcutGroundingDinoDetectInput = document.getElementById("shortcutGroundingDinoDetectInput");
const shortcutClassifyInput = document.getElementById("shortcutClassifyInput");
const shortcutClassifyModelInput = document.getElementById("shortcutClassifyModelInput");
const shortcutDeleteBoxInput = document.getElementById("shortcutDeleteBoxInput");
const shortcutClearBoxesInput = document.getElementById("shortcutClearBoxesInput");
const shortcutReloadAnnotationInput = document.getElementById("shortcutReloadAnnotationInput");
const shortcutCancelSelectInput = document.getElementById("shortcutCancelSelectInput");

const saveShortcutConfigBtn = document.getElementById("saveShortcutConfigBtn");
const resetShortcutConfigBtn = document.getElementById("resetShortcutConfigBtn");
const shortcutRejectImageInput = document.getElementById("shortcutRejectImageInput");
const sampledColorSwatch = document.getElementById("sampledColorSwatch");
const sampledRgbText = document.getElementById("sampledRgbText");
const classifyRulesList = document.getElementById("classifyRulesList");
const saveClassifyRulesBtn = document.getElementById("saveClassifyRulesBtn");
const reloadClassifyRulesBtn = document.getElementById("reloadClassifyRulesBtn");

let allImages = [];
let allAnnotations = {};
let filteredImages = [];
let isFilterDefaultClassOnly = false;

let currentImageName = null;
let currentImage = null;

let boxes = [];
let classBoxes = [];
let boxInfos = [];
let selectedBoxIndex = -1;
let classifyRules = [];
let isPickColorMode = false;
let canvasClassMenuBoxIndex = -1;
let lastSelectBoxShortcutIndex = -1;
let lastSelectBoxShortcutTime = 0;

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
    prompt: "tennis ball. blue bean bag. red bean bag. white teddy bear. brown teddy bear",
    boxThreshold: 0.18,
    textThreshold: 0.20,
    useNms: true,
    nmsIouThreshold: 0.20
};

const defaultShortcutConfig = {
    previousImage: "a",
    nextImage: "d",
    saveXml: "s",
    yoloDetect: "y",
    groundingDinoDetect: "g",
    classifyPureColor: "f",
    classifyModel: "m",
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

    groundingPromptInput.value = groundingDinoDefaultConfig.prompt;
    groundingBoxThresholdInput.value = String(groundingDinoDefaultConfig.boxThreshold);
    groundingTextThresholdInput.value = String(groundingDinoDefaultConfig.textThreshold);
    groundingUseNmsInput.checked = groundingDinoDefaultConfig.useNms;
    groundingNmsIouInput.value = String(groundingDinoDefaultConfig.nmsIouThreshold);

    groundingBoxThresholdInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") e.target.blur();
    });
    groundingTextThresholdInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") e.target.blur();
    });
}

async function fetchGroundingDinoDefaults() {
    try {
        const response = await fetch("/api/grounding_dino_defaults");
        const data = await response.json();

        if (data.success && data.defaults) {
            groundingDinoDefaultConfig.prompt = data.defaults.prompt || groundingDinoDefaultConfig.prompt;
            groundingDinoDefaultConfig.boxThreshold = data.defaults.box_threshold ?? groundingDinoDefaultConfig.boxThreshold;
            groundingDinoDefaultConfig.textThreshold = data.defaults.text_threshold ?? groundingDinoDefaultConfig.textThreshold;
            groundingDinoDefaultConfig.useNms = data.defaults.use_nms ?? groundingDinoDefaultConfig.useNms;
            groundingDinoDefaultConfig.nmsIouThreshold = data.defaults.nms_iou_threshold ?? groundingDinoDefaultConfig.nmsIouThreshold;

            initGroundingDinoInputs();
        }
    } catch (error) {
        console.warn("无法从后端获取 Grounding DINO 默认配置，使用内置默认值", error);
    }
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


async function fetchShortcutConfig() {
    try {
        const response = await fetch("/api/shortcut_config");
        const data = await response.json();

        if (!data.success || !data.config) {
            return false;
        }

        shortcutConfig = {
            ...defaultShortcutConfig,
            ...data.config,
            selectBoxKeys: defaultShortcutConfig.selectBoxKeys
        };
        saveShortcutConfig(shortcutConfig);
        fillShortcutConfigInputs();
        return true;
    } catch (error) {
        console.warn("读取后端快捷键配置失败，使用本地配置", error);
        return false;
    }
}


async function saveShortcutConfigToBackend(config) {
    const response = await fetch("/api/shortcut_config", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ config })
    });

    const data = await response.json();

    if (!data.success) {
        throw new Error(data.message || "Save shortcut config failed");
    }

    return data.config || config;
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
            key: "classifyPureColor",
            input: shortcutClassifyInput,
            defaultValue: defaultShortcutConfig.classifyPureColor
        },
        {
            key: "classifyModel",
            input: shortcutClassifyModelInput,
            defaultValue: defaultShortcutConfig.classifyModel
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


function getCurrentShortcutConfig() {
    const bindings = getShortcutInputBindings();
    const currentConfig = {
        ...shortcutConfig,
        selectBoxKeys: defaultShortcutConfig.selectBoxKeys
    };

    for (const item of bindings) {
        // 空值表示"无快捷键"(按 Backspace 清除),不回退到默认值
        const inputValue = item.input.value.trim();
        currentConfig[item.key] = inputValue === "" ? "" : normalizeShortcutKey(inputValue);
    }

    return currentConfig;
}


function bindShortcutInputCapture() {
    const bindings = getShortcutInputBindings();

    for (const item of bindings) {
        // 空输入框显示提示: Backspace 表示该功能无快捷键
        item.input.placeholder = "无快捷键(按Backspace)";

        item.input.addEventListener("keydown", (event) => {
            if (event.key === "Tab") {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            if (event.key === "Backspace") {
                item.input.value = "";
                statusTextEl.textContent = `状态：已清除 ${item.key} 的快捷键（该功能不再绑定按键）`;
                return;
            }

            const key = normalizeShortcutKey(event.key);

            item.input.value = key;
            statusTextEl.textContent = `状态：已设置 ${item.key} 快捷键为 ${key}，当前已生效，保存后下次打开仍生效`;
        });

        item.input.addEventListener("mousedown", event => {
            event.preventDefault();
            item.input.focus();
        });

        item.input.addEventListener("click", event => {
            event.preventDefault();
            item.input.focus();
            statusTextEl.textContent = "状态：请按下想设置的按键（按 Backspace 表示该功能无快捷键）";
        });
    }
}


async function readShortcutConfigInputs() {
    const bindings = getShortcutInputBindings();

    const nextConfig = {
        ...shortcutConfig,
        selectBoxKeys: defaultShortcutConfig.selectBoxKeys
    };

    for (const item of bindings) {
        // 空值表示"无快捷键"(按 Backspace 清除),不回退到默认值
        const inputValue = item.input.value.trim();
        const value = inputValue === "" ? "" : normalizeShortcutKey(inputValue);
        nextConfig[item.key] = value;
    }

    const conflictMessage = findShortcutConflictMessage(nextConfig);

    if (conflictMessage) {
        statusTextEl.textContent = conflictMessage;
        return;
    }

    shortcutConfig = nextConfig;
    try {
        const savedConfig = await saveShortcutConfigToBackend(shortcutConfig);
        shortcutConfig = {
            ...defaultShortcutConfig,
            ...savedConfig,
            selectBoxKeys: defaultShortcutConfig.selectBoxKeys
        };
        saveShortcutConfig(shortcutConfig);
        fillShortcutConfigInputs();
        statusTextEl.textContent = "状态：快捷键配置已保存到后端";
    } catch (error) {
        console.error("快捷键配置保存到后端失败：", error);
        saveShortcutConfig(shortcutConfig);
        fillShortcutConfigInputs();
        statusTextEl.textContent = "状态：后端保存失败，已临时保存到浏览器";
    }
}


async function resetShortcutConfig() {
    const success = await fetchShortcutConfig();

    if (success) {
        statusTextEl.textContent = "状态：已恢复后端保存的快捷键配置";
        return;
    }

    shortcutConfig = loadShortcutConfig();
    fillShortcutConfigInputs();
    statusTextEl.textContent = "状态：后端读取失败，已恢复浏览器保存的快捷键配置";
}


function findShortcutConflictMessage(config) {
    const keys = [
        config.previousImage,
        config.nextImage,
        config.saveXml,
        config.yoloDetect,
        config.groundingDinoDetect,
        config.classifyPureColor,
        config.classifyModel,
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

        // 空值表示"无快捷键",不参与重复检查(多个功能可以同时无快捷键)
        if (!normalizedKey) {
            continue;
        }

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
        .filter(key => key && reserved.includes(key));

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

    if (currentImageName) {
        scrollPreviewWallToImage(currentImageName);
    }
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

        const rawImages = data.images || [];
        allAnnotations = {};
        allImages = rawImages.map(item => {
            if (typeof item === "string") {
                allAnnotations[item] = { boxes: [], class_boxes: [] };
                return item;
            }
            const nextClassBoxes = item.class_boxes || [];
            allAnnotations[item.name] = {
                boxes: classBoxesToBoxes(nextClassBoxes),
                class_boxes: nextClassBoxes
            };
            return item.name;
        });
        renderImageList();

        if (isPreviewWallVisible()) {
            renderPreviewWall();
        }

        statusTextEl.textContent = "状态：图片列表加载完成";

        const lastImageName = localStorage.getItem("lastImageName");
        if (lastImageName && allImages.includes(lastImageName)) {
            selectImage(lastImageName);
        } else if (allImages.length > 0) {
            selectImage(allImages[0]);
        }
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
        yoloModelCount = data.models.length;

        if (data.default_model !== undefined && data.default_model !== null) {
            currentYoloModelId = String(data.default_model);
        } else if (data.models.length > 0) {
            currentYoloModelId = String(data.models[0].id);
        }

        for (const model of data.models) {
            const li = document.createElement("li");
            const shortcutHint = yoloModelCount <= 9
                ? ` (Ctrl+${model.id})`
                : "";
            li.textContent = model.name + shortcutHint;
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

        if (yoloModelCount > 1) {
            statusTextEl.textContent = `状态：已加载 ${yoloModelCount} 个 YOLO 模型，Ctrl+1~Ctrl+${yoloModelCount} 快捷切换`;
        } else {
            statusTextEl.textContent = "状态：YOLO 模型列表加载完成";
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
        if (keyword && !imageName.toLowerCase().includes(keyword)) {
            return false;
        }

        if (isFilterDefaultClassOnly && !imageHasDefaultClass(imageName)) {
            return false;
        }

        return true;
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


function imageHasDefaultClass(imageName) {
    const annotation = allAnnotations[imageName] || {};
    const nextClassBoxes = annotation.class_boxes || [];

    for (const classBox of nextClassBoxes) {
        if (Array.isArray(classBox) && classBox[0] === "object") {
            return true;
        }
    }

    return false;
}


function toggleDefaultClassFilter() {
    isFilterDefaultClassOnly = !isFilterDefaultClassOnly;
    updateDefaultClassFilterButton();

    renderImageList();
}


function updateDefaultClassFilterButton() {
    if (filterDefaultClassBtn) {
        filterDefaultClassBtn.classList.toggle("active", isFilterDefaultClassOnly);

        if (isFilterDefaultClassOnly) {
            filterDefaultClassBtn.textContent = "\u53d6\u6d88\u7b5b\u9009";
            filterDefaultClassBtn.title = "\u53d6\u6d88\u672a\u5206\u7c7b\u7b5b\u9009";
        } else {
            filterDefaultClassBtn.textContent = "\u7b5b\u9009\u672a\u5206\u7c7b";
            filterDefaultClassBtn.title = "\u7b5b\u9009\u672a\u5206\u7c7b\u56fe\u7247";
        }
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

function scrollPreviewWallToImage(imageName) {
    const targetIndex = previewWallFiltered.indexOf(imageName);
    if (targetIndex === -1) return;

    if (previewWallObserver) {
        previewWallObserver.disconnect();
        previewWallObserver = null;
    }

    if (previewWallSentinel && previewWallSentinel.parentNode) {
        previewWallSentinel.remove();
    }
    previewWallSentinel = null;

    while (previewWallLoadedCount <= targetIndex) {
        const startIndex = previewWallLoadedCount;
        const endIndex = Math.min(
            startIndex + PREVIEW_WALL_BATCH_SIZE,
            previewWallFiltered.length
        );
        const batch = previewWallFiltered.slice(startIndex, endIndex);

        for (const imgName of batch) {
            const item = createPreviewWallItem(imgName);
            previewWallGrid.appendChild(item);
            const canvas = item.querySelector("canvas");
            drawThumbnail(imgName, canvas);
        }

        previewWallLoadedCount = endIndex;

        if (previewWallLoadedCount >= previewWallFiltered.length) break;
    }

    previewWallSentinel = createPreviewWallSentinel();
    previewWallGrid.appendChild(previewWallSentinel);
    setupPreviewWallObserver();

    requestAnimationFrame(() => {
        const items = previewWallGrid.querySelectorAll(".preview-wall-item");
        for (const item of items) {
            if (item.dataset.imageName === imageName) {
                item.scrollIntoView({ block: "center", behavior: "smooth" });
                break;
            }
        }
    });
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

        const annot = allAnnotations[imageName];
        const boxes = annot ? annot.boxes : null;
        if (boxes && boxes.length > 0) {
            previewCtx.strokeStyle = canvasColors.previewBoxStroke;
            previewCtx.lineWidth = 1;
            for (const box of boxes) {
                const bx = offsetX + box[0] * scale;
                const by = offsetY + box[1] * scale;
                const bw = (box[2] - box[0]) * scale;
                const bh = (box[3] - box[1]) * scale;
                previewCtx.strokeRect(bx, by, bw, bh);
            }
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
    hideCanvasClassMenu();
    currentImageName = imageName;
    boxes = [];
    classBoxes = [];
    boxInfos = [];
    selectedBoxIndex = -1;

    try {
        localStorage.setItem("lastImageName", imageName);
    } catch (e) {}

    updatePositionProgress(imageName);
    if (jumpToImageInput) {
        jumpToImageInput.value = imageName;
    }

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

function updatePositionProgress(imageName) {
    if (!positionProgressEl) return;
    const total = allImages.length;
    if (total === 0) {
        positionProgressEl.textContent = "- / -";
        return;
    }
    const index = allImages.indexOf(imageName);
    positionProgressEl.textContent = (index >= 0 ? index + 1 : "?") + " / " + total;
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
        const info = boxInfos[i] || makeDefaultBoxInfo();
        const classColor = getClassMiddleColor(info.name);

        const isSelected = i === selectedBoxIndex;

        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeStyle = isSelected ? canvasColors.selectedBox : canvasColors.normalBox;

        ctx.strokeRect(
            canvasBox.x,
            canvasBox.y,
            canvasBox.w,
            canvasBox.h
        );

        if (classColor) {
            ctx.save();
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeStyle = rgbToCss(classColor);
            ctx.beginPath();
            ctx.moveTo(canvasBox.x, canvasBox.y);
            ctx.lineTo(canvasBox.x + canvasBox.w, canvasBox.y + canvasBox.h);
            ctx.moveTo(canvasBox.x + canvasBox.w, canvasBox.y);
            ctx.lineTo(canvasBox.x, canvasBox.y + canvasBox.h);
            ctx.stroke();
            ctx.restore();
        }

        const labelText = ` ${i + 1}`;

        ctx.font = "bold 22px Arial";

        const labelPaddingX = 0;
        const labelHeight = 28;

        const labelX = canvasBox.x;
        const labelY = Math.max(canvasBox.y - labelHeight, 0);

        ctx.fillStyle = canvasColors.labelText;
        ctx.fillText(
            labelText,
            labelX + labelPaddingX,
            labelY + 22
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


function showCanvasClassMenu(index) {
    if (!canvasClassMenu || index < 0 || index >= boxes.length) {
        return;
    }

    selectedBoxIndex = index;
    canvasClassMenuBoxIndex = index;

    const canvasBox = imageBoxToCanvasBox(boxes[index]);
    const classNames = getClassNamesFromRules();

    canvasClassMenu.innerHTML = "";
    canvasClassMenu.classList.remove("hidden");
    canvasClassMenu.style.left = `${Math.max(0, canvasBox.x + canvasBox.w + 8)}px`;
    canvasClassMenu.style.top = `${Math.max(0, canvasBox.y)}px`;

    const title = document.createElement("div");
    title.classList.add("canvas-class-menu-title");
    title.textContent = `box ${index + 1}`;
    canvasClassMenu.appendChild(title);

    classNames.forEach((name, classIndex) => {
        const item = document.createElement("button");
        item.type = "button";
        item.classList.add("canvas-class-menu-item");
        item.dataset.classIndex = String(classIndex + 1);
        item.dataset.className = name;
        item.textContent = `${classIndex + 1}. ${name}`;
        item.addEventListener("click", event => {
            event.stopPropagation();
            chooseCanvasClassByIndex(classIndex + 1);
        });
        canvasClassMenu.appendChild(item);
    });

    renderBoxList();
    redrawCanvas();
}


function hideCanvasClassMenu() {
    if (!canvasClassMenu) {
        return;
    }

    canvasClassMenu.classList.add("hidden");
    canvasClassMenu.innerHTML = "";
    canvasClassMenuBoxIndex = -1;
}


function chooseCanvasClassByIndex(classNumber) {
    if (canvasClassMenuBoxIndex < 0 || canvasClassMenuBoxIndex >= boxes.length) {
        hideCanvasClassMenu();
        return false;
    }

    const classNames = getClassNamesFromRules();
    const className = classNames[classNumber - 1];

    if (!className) {
        return false;
    }

    updateBoxClassName(canvasClassMenuBoxIndex, className);
    hideCanvasClassMenu();
    return true;
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


function classBoxesToBoxes(nextClassBoxes) {
    const nextBoxes = [];

    for (const classBox of nextClassBoxes || []) {
        if (Array.isArray(classBox) && classBox.length === 5) {
            nextBoxes.push(classBox.slice(1));
        }
    }

    return nextBoxes;
}


function classBoxesToBoxInfos(nextClassBoxes) {
    const nextBoxInfos = [];

    for (const classBox of nextClassBoxes || []) {
        if (Array.isArray(classBox) && classBox.length === 5) {
            nextBoxInfos.push({
                name: classBox[0] || "object",
                score: null
            });
        }
    }

    return nextBoxInfos;
}


function boxesToClassBoxes(nextBoxes, nextBoxInfos) {
    return (nextBoxes || []).map((box, index) => {
        const info = nextBoxInfos[index] || makeDefaultBoxInfo();
        return [info.name || "object", ...box];
    });
}


function getClassNamesFromRules() {
    const names = ["object"];

    for (const rule of classifyRules || []) {
        const name = rule.name || "";

        if (name && !names.includes(name)) {
            names.push(name);
        }
    }

    return names;
}


function getClassRuleByName(name) {
    for (const rule of classifyRules || []) {
        if (rule.name === name) {
            return rule;
        }
    }

    return null;
}


function getClassMiddleColor(name) {
    const rule = getClassRuleByName(name);

    if (!rule) {
        return null;
    }

    const rgbMin = normalizeRgbArray(rule.rgb_min);
    const rgbMax = normalizeRgbArray(rule.rgb_max);

    return rgbMin.map((value, index) => {
        return Math.round((value + rgbMax[index]) / 2);
    });
}


function getClassDisplayColor(name) {
    return getClassMiddleColor(name) || [128,128,128];
}


function makeClassSelectOptions(currentName) {
    return getClassNamesFromRules().map(name => {
        const selected = name === currentName ? " selected" : "";
        return `<option value="${escapeHtml(name)}"${selected}>${escapeHtml(name)}</option>`;
    }).join("");
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
        const classColor = getClassDisplayColor(info.name);

        li.innerHTML = `
            <div class="box-list-title">
                <span class="box-class-main">
                    <span class="box-class-swatch" style="background:${rgbToCss(classColor)}"></span>
                    <span>${i + 1}.</span>
                    <select class="box-class-select" data-index="${i}">
                        ${makeClassSelectOptions(info.name)}
                    </select>
                </span>
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

    bindBoxClassSelects();
}


function escapeHtml(text) {
    return String(text)
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;")
        .replaceAll('"',"&quot;")
        .replaceAll("'","&#039;");
}


function bindBoxClassSelects() {
    const selects = boxListEl.querySelectorAll(".box-class-select");

    for (const select of selects) {
        select.addEventListener("click", event => {
            event.stopPropagation();
        });

        select.addEventListener("change", event => {
            const index = Number(event.target.dataset.index);
            updateBoxClassName(index, event.target.value);
        });
    }
}


function updateBoxClassName(index, className) {
    if (index < 0 || index >= boxes.length) {
        return;
    }

    if (!boxInfos[index]) {
        boxInfos[index] = makeDefaultBoxInfo();
    }

    boxInfos[index].name = className || "object";
    classBoxes = boxesToClassBoxes(boxes, boxInfos);

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = `\u72b6\u6001\uff1a\u7b2c ${index + 1} \u4e2a\u6846\u7c7b\u522b\u5df2\u6539\u4e3a ${className || "object"}`;
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

    classBoxes = boxesToClassBoxes(boxes, boxInfos);

    const payload = {
        image_name: currentImageName,
        class_boxes: classBoxes
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

        allAnnotations[currentImageName] = {
            boxes: boxes.slice(),
            class_boxes: classBoxes.slice()
        };

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

    removeRejectedImageFromView(imageNameToReject, currentIndex);

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
        delete allAnnotations[imageNameToReject];

        boxes = [];
        classBoxes = [];
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
            if (jumpToImageInput) {
                jumpToImageInput.value = "";
            }
            if (positionProgressEl) {
                positionProgressEl.textContent = "- / -";
            }
        }

        if (typeof renderPreviewWall === "function" && isPreviewWallVisible()) {
            renderPreviewWall();
        }
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：移入废弃区失败，无法连接后端";
    }
}


function removeRejectedImageFromView(imageNameToReject, currentIndex) {
    allImages = allImages.filter(name => name !== imageNameToReject);
    filteredImages = filteredImages.filter(name => name !== imageNameToReject);
    delete allAnnotations[imageNameToReject];

    boxes = [];
    classBoxes = [];
    boxInfos = [];
    selectedBoxIndex = -1;
    currentImage = null;
    currentImageName = null;

    renderImageList();
    renderBoxList();
    redrawCanvas();

    if (filteredImages.length > 0) {
        let nextIndex = currentIndex;

        if (nextIndex >= filteredImages.length) {
            nextIndex = filteredImages.length - 1;
        }

        selectImage(filteredImages[nextIndex]);
    } else {
        currentImageNameEl.textContent = "\u5f53\u524d\u56fe\u7247\uff1a\u65e0";
        if (jumpToImageInput) {
            jumpToImageInput.value = "";
        }
        if (positionProgressEl) {
            positionProgressEl.textContent = "- / -";
        }
    }

    if (typeof renderPreviewWall === "function" && isPreviewWallVisible()) {
        renderPreviewWall();
    }
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

        classBoxes = result.class_boxes || [];
        boxes = classBoxesToBoxes(classBoxes);
        boxInfos = classBoxesToBoxInfos(classBoxes);
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

        if (Array.isArray(data.class_boxes) && data.class_boxes.length > 0) {
            classBoxes = data.class_boxes;
            boxes = classBoxesToBoxes(classBoxes);
            boxInfos = classBoxesToBoxInfos(classBoxes);
        } else if (Array.isArray(data.objects) && data.objects.length > 0) {
            const normalized = normalizeObjectsToBoxesAndInfos(data.objects);

            boxes = normalized.boxes;
            boxInfos = normalized.boxInfos;
            classBoxes = boxesToClassBoxes(boxes, boxInfos);
        } else {
            boxes = [];
            classBoxes = [];
            boxInfos = [];
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



async function classifyCurrentBoxes(classifier = "pure_color") {
    if (!currentImageName) {
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u56fe\u7247";
        return false;
    }

    if (boxes.length === 0) {
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u5f53\u524d\u6ca1\u6709\u6846\u53ef\u5206\u7c7b";
        return false;
    }

    classBoxes = boxesToClassBoxes(boxes, boxInfos);

    const payload = {
        image_name: currentImageName,
        class_boxes: classBoxes,
        classifier: classifier
    };

    if (classifier === "pure_color") {
        try {
            payload.rules = classifyRulesList ? collectClassifyRulesFromForm() : classifyRules;
        } catch (error) {
            statusTextEl.textContent = `\u72b6\u6001\uff1a\u5206\u7c7b\u9608\u503c\u65e0\u6548\uff0c${error.message}`;
            return false;
        }
    } else {
        payload.model_select = currentClassifyModelId;
    }

    const actionName = classifier === "model" ? "模型分类" : "纯色分类";
    statusTextEl.textContent = `\u72b6\u6001\uff1a\u6b63\u5728${actionName}当前框...`;

    try {
        const response = await fetch("/api/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `\u72b6\u6001\uff1a\u5206\u7c7b\u5931\u8d25\uff0c${data.message}`;
            return false;
        }

        classBoxes = data.class_boxes || [];
        boxes = classBoxesToBoxes(classBoxes);
        boxInfos = classBoxesToBoxInfos(classBoxes);
        selectedBoxIndex = -1;

        renderBoxList();
        redrawCanvas();
        canvas.focus();

        statusTextEl.textContent = `\u72b6\u6001\uff1a${actionName}完成，框数量 ${boxes.length}`;
        return true;
    } catch (error) {
        console.error("\u5206\u7c7b\u8bf7\u6c42\u5931\u8d25\uff1a", error);
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u5206\u7c7b\u5931\u8d25\uff0c\u65e0\u6cd5\u8fde\u63a5\u540e\u7aef";
        return false;
    }
}

async function loadClassifyModels() {
    if (!classifyModelDropdown || !modelClassifyBtn) {
        return;
    }

    try {
        const response = await fetch("/api/classify_models");
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = "状态：加载分类模型列表失败";
            return;
        }

        classifyModelDropdown.innerHTML = "";
        classifyModelCount = Array.isArray(data.models) ? data.models.length : 0;

        if (data.default_model !== undefined && data.default_model !== null) {
            currentClassifyModelId = String(data.default_model);
        } else if (data.models.length > 0) {
            currentClassifyModelId = String(data.models[0].id);
        }

        for (const model of data.models || []) {
            const li = document.createElement("li");
            li.textContent = model.name;
            li.dataset.modelId = String(model.id);

            if (String(model.id) === currentClassifyModelId) {
                li.classList.add("selected");
                modelClassifyBtn.textContent = `模型分类 ${model.id}`;
            }

            li.addEventListener("click", (event) => {
                event.stopPropagation();
                currentClassifyModelId = String(model.id);
                modelClassifyBtn.textContent = `模型分类 ${model.id}`;
                classifyModelDropdown.querySelectorAll("li").forEach(item => {
                    item.classList.toggle("selected", item.dataset.modelId === currentClassifyModelId);
                });
                closeClassifyModelDropdown();
            });

            classifyModelDropdown.appendChild(li);
        }
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：无法加载分类模型列表";
    }
}

function toggleClassifyModelDropdown() {
    if (!classifyModelDropdown) {
        return;
    }

    classifyModelDropdown.classList.toggle("hidden");
}

function closeClassifyModelDropdown() {
    if (!classifyModelDropdown) {
        return;
    }

    classifyModelDropdown.classList.add("hidden");
}

async function loadClassifyRules() {
    if (!classifyRulesList) {
        return;
    }

    try {
        const response = await fetch("/api/classify_rules");
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `\u72b6\u6001\uff1a\u8bfb\u53d6\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff0c${data.message}`;
            return;
        }

        classifyRules = data.rules || [];
        renderClassifyRules();
        renderBoxList();
        redrawCanvas();
    } catch (error) {
        console.error("\u8bfb\u53d6\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff1a", error);
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u8bfb\u53d6\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff0c\u65e0\u6cd5\u8fde\u63a5\u540e\u7aef";
    }
}


function renderClassifyRules() {
    if (!classifyRulesList) {
        return;
    }

    classifyRulesList.innerHTML = "";
    const header = document.createElement("div");
    header.classList.add("classify-rule-header");
    header.innerHTML = `
        <div>class / RGB range</div>
    `;
    classifyRulesList.appendChild(header);

    for (let i = 0; i < classifyRules.length; i++) {
        const rule = classifyRules[i];
        const name = rule.name || "object";

        if (name === "object") {
            continue;
        }

        const rgbMin = normalizeRgbArray(rule.rgb_min);
        const rgbMax = normalizeRgbArray(rule.rgb_max);

        const item = document.createElement("div");
        item.classList.add("classify-rule-item");
        item.dataset.index = String(i);
        item.dataset.name = name;

        item.innerHTML = `
            <div class="rule-class-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
            <div class="rule-rgb-row">
                <span class="rgb-swatch" style="background:${rgbToCss(rgbMin)}"></span>
                <span class="rule-rgb-label">min</span>
                ${makeRgbInputs("min", rgbMin)}
            </div>
            <div class="rule-rgb-row">
                <span class="rgb-swatch" style="background:${rgbToCss(rgbMax)}"></span>
                <span class="rule-rgb-label">max</span>
                ${makeRgbInputs("max", rgbMax)}
            </div>
        `;

        classifyRulesList.appendChild(item);
    }

    bindRgbRuleInputs();
}


function makeRgbInputs(kind, rgb) {
    return ["r","g","b"].map((channel, index) => {
        return `<input class="rgb-input" type="number" min="0" max="255" step="1" value="${rgb[index]}" data-kind="${kind}" data-channel="${channel}">`;
    }).join("");
}


function bindRgbRuleInputs() {
    const inputs = classifyRulesList.querySelectorAll(".rgb-input");

    for (const input of inputs) {
        input.addEventListener("input", () => {
            input.value = String(clamp(Number(input.value) || 0, 0, 255));
            updateRuleSwatches(input.closest(".classify-rule-item"));
        });
    }
}


function updateRuleSwatches(item) {
    if (!item) {
        return;
    }

    const minRgb = readRgbInputsByKind(item, "min");
    const maxRgb = readRgbInputsByKind(item, "max");
    const swatches = item.querySelectorAll(".rgb-swatch");

    if (swatches[0]) {
        swatches[0].style.background = rgbToCss(minRgb);
    }
    if (swatches[1]) {
        swatches[1].style.background = rgbToCss(maxRgb);
    }
}


function collectClassifyRulesFromForm() {
    const nextRules = [];
    const items = classifyRulesList.querySelectorAll(".classify-rule-item");

    for (const item of items) {
        const name = item.dataset.name || "";

        if (!name) {
            throw new Error("\u7c7b\u522b\u540d\u4e0d\u80fd\u4e3a\u7a7a");
        }

        nextRules.push({
            name: name,
            rgb_min: readRgbInputsByKind(item, "min"),
            rgb_max: readRgbInputsByKind(item, "max")
        });
    }

    return nextRules;
}


function readRgbInputsByKind(item, kind) {
    return ["r","g","b"].map(channel => {
        const input = item.querySelector(`.rgb-input[data-kind="${kind}"][data-channel="${channel}"]`);
        return clamp(Number(input ? input.value : 0) || 0, 0, 255);
    });
}


async function saveClassifyRules() {
    try {
        const rules = collectClassifyRulesFromForm();

        const response = await fetch("/api/classify_rules", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ rules })
        });

        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = `\u72b6\u6001\uff1a\u4fdd\u5b58\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff0c${data.message}`;
            return;
        }

        classifyRules = data.rules || rules;
        renderClassifyRules();
        renderBoxList();
        redrawCanvas();
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u5206\u7c7b\u9608\u503c\u5df2\u4fdd\u5b58";
    } catch (error) {
        console.error("\u4fdd\u5b58\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff1a", error);
        statusTextEl.textContent = `\u72b6\u6001\uff1a\u4fdd\u5b58\u5206\u7c7b\u9608\u503c\u5931\u8d25\uff0c${error.message}`;
    }
}


function normalizeRgbArray(rgb) {
    if (!Array.isArray(rgb) || rgb.length !== 3) {
        return [0,0,0];
    }

    return rgb.map(value => clamp(Number(value) || 0, 0, 255));
}


function rgbToCss(rgb) {
    const nextRgb = normalizeRgbArray(rgb);
    return `rgb(${nextRgb[0]}, ${nextRgb[1]}, ${nextRgb[2]})`;
}


function togglePickColorMode() {
    isPickColorMode = !isPickColorMode;

    if (pickColorBtn) {
        pickColorBtn.classList.toggle("active", isPickColorMode);
    }

    statusTextEl.textContent = isPickColorMode
        ? "\u72b6\u6001\uff1a\u53d6\u8272\u6a21\u5f0f\u5df2\u5f00\u542f\uff0c\u8bf7\u70b9\u51fb\u56fe\u50cf\u5185\u90e8\u50cf\u7d20"
        : "\u72b6\u6001\uff1a\u53d6\u8272\u6a21\u5f0f\u5df2\u5173\u95ed";
}


function sampleRgbAtCanvasEvent(event) {
    if (!currentImage) {
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u56fe\u7247";
        return;
    }

    resizeCanvasToDisplaySize();
    updateImageView();

    const point = getMousePosition(event);

    if (!isPointInsideDisplayedImage(point)) {
        statusTextEl.textContent = "\u72b6\u6001\uff1a\u8bf7\u70b9\u51fb\u56fe\u50cf\u5185\u90e8\u53d6\u8272";
        return;
    }

    const imagePoint = canvasPointToImagePoint(point);
    const rgb = sampleImageRgb(imagePoint.x, imagePoint.y);
    updateSampledRgbDisplay(rgb, imagePoint);
    redrawCanvas();
}


function sampleImageRgb(x, y) {
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = currentImage.naturalWidth;
    sampleCanvas.height = currentImage.naturalHeight;

    const sampleCtx = sampleCanvas.getContext("2d");
    sampleCtx.drawImage(currentImage, 0, 0);

    const pixel = sampleCtx.getImageData(
        clamp(x, 0, currentImage.naturalWidth - 1),
        clamp(y, 0, currentImage.naturalHeight - 1),
        1,
        1
    ).data;

    return [pixel[0], pixel[1], pixel[2]];
}


function updateSampledRgbDisplay(rgb, imagePoint) {
    if (sampledColorSwatch) {
        sampledColorSwatch.style.background = rgbToCss(rgb);
    }

    if (sampledRgbText) {
        sampledRgbText.textContent = `RGB: ${rgb[0]}, ${rgb[1]}, ${rgb[2]}  x:${imagePoint.x} y:${imagePoint.y}`;
    }

    statusTextEl.textContent = `\u72b6\u6001\uff1a\u53d6\u8272 RGB(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
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
    classBoxes.splice(selectedBoxIndex, 1);
    boxInfos.splice(selectedBoxIndex, 1);
    hideCanvasClassMenu();

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
    classBoxes = [];
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

    if (isPickColorMode) {
        sampleRgbAtCanvasEvent(event);
        return;
    }

    resizeCanvasToDisplaySize();
    updateImageView();

    const point = getMousePosition(event);

    if (!isPointInsideDisplayedImage(point)) {
        return;
    }

    hideCanvasClassMenu();
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
        classBoxes = boxesToClassBoxes(boxes, boxInfos);
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

if (jumpToImageInput) {
    jumpToImageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            const targetName = jumpToImageInput.value.trim();
            if (targetName && allImages.includes(targetName)) {
                selectImage(targetName);
            } else {
                statusTextEl.textContent = "状态：图片 " + targetName + " 不存在";
            }
        }
    });
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
if (filterDefaultClassBtn) {
    filterDefaultClassBtn.addEventListener("click", toggleDefaultClassFilter);
}
if (pureColorClassifyBtn) {
    pureColorClassifyBtn.addEventListener("click", () => classifyCurrentBoxes("pure_color"));
}
if (modelClassifyBtn) {
    modelClassifyBtn.addEventListener("click", () => classifyCurrentBoxes("model"));
}
if (classifyModelDropdownBtn) {
    classifyModelDropdownBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleClassifyModelDropdown();
    });
}
if (pickColorBtn) {
    pickColorBtn.addEventListener("click", togglePickColorMode);
}
if (saveClassifyRulesBtn) {
    saveClassifyRulesBtn.addEventListener("click", saveClassifyRules);
}
if (reloadClassifyRulesBtn) {
    reloadClassifyRulesBtn.addEventListener("click", loadClassifyRules);
}
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

document.addEventListener("click", (event) => {
    if (!classifyModelDropdown || !classifyModelDropdownBtn) {
        return;
    }
    if (!classifyModelDropdownBtn.contains(event.target) && !classifyModelDropdown.contains(event.target)) {
        closeClassifyModelDropdown();
    }
});

document.addEventListener("click", (event) => {
    if (!classifyModelDropdown || !classifyModelDropdownBtn) {
        return;
    }
    if (!classifyModelDropdownBtn.contains(event.target) && !classifyModelDropdown.contains(event.target)) {
        closeClassifyModelDropdown();
    }
});

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

    if (canvasClassMenuBoxIndex !== -1) {
        const classNumber = parseInt(event.key, 10);

        if (classNumber >= 1 && classNumber <= 9) {
            if (chooseCanvasClassByIndex(classNumber)) {
                event.preventDefault();
                return;
            }
        }

        if (event.key === "Escape") {
            event.preventDefault();
            hideCanvasClassMenu();
            return;
        }
    }

    const activeShortcutConfig = getCurrentShortcutConfig();

    const selectBoxIndex = activeShortcutConfig.selectBoxKeys.findIndex(shortcut => {
        return eventMatchesShortcut(event, shortcut);
    });

    if (selectBoxIndex !== -1) {
        event.preventDefault();
        const now = Date.now();

        if (
            selectedBoxIndex === selectBoxIndex &&
            lastSelectBoxShortcutIndex === selectBoxIndex &&
            now - lastSelectBoxShortcutTime <= 450
        ) {
            showCanvasClassMenu(selectBoxIndex);
            lastSelectBoxShortcutIndex = -1;
            lastSelectBoxShortcutTime = 0;
        } else {
            selectBoxByIndex(selectBoxIndex);
            hideCanvasClassMenu();
            lastSelectBoxShortcutIndex = selectBoxIndex;
            lastSelectBoxShortcutTime = now;
        }
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.cancelSelect)) {
        event.preventDefault();
        cancelSelectedBox();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.saveXml)) {
        event.preventDefault();
        await saveXml();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.classifyPureColor)) {
        event.preventDefault();
        await classifyCurrentBoxes("pure_color");
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.classifyModel)) {
        event.preventDefault();
        await classifyCurrentBoxes("model");
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.previousImage)) {
        event.preventDefault();
        selectPreviousImage();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.nextImage)) {
        event.preventDefault();
        selectNextImage();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.yoloDetect)) {
        event.preventDefault();
        autoDetect("yolo", currentYoloModelId);
        return;
    }

    if (event.ctrlKey && !event.altKey && !event.metaKey && yoloModelCount > 0) {
        const digit = parseInt(event.key, 10);
        if (digit >= 1 && digit <= yoloModelCount) {
            event.preventDefault();
            autoDetect("yolo", String(digit));
            return;
        }
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.groundingDinoDetect)) {
        event.preventDefault();
        autoDetect("grounding_dino");
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.deleteBox)) {
        event.preventDefault();
        deleteSelectedBox();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.clearBoxes)) {
        event.preventDefault();
        clearAllBoxes();
        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.reloadAnnotation)) {
        event.preventDefault();

        if (currentImageName) {
            loadAnnotation(currentImageName);
        }

        return;
    }

    if (eventMatchesShortcut(event, activeShortcutConfig.rejectImage)) {
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
fetchGroundingDinoDefaults();
loadClassifyRules();
loadClassifyModels();
fillShortcutConfigInputs();
bindShortcutInputCapture();
updateDefaultClassFilterButton();
fetchShortcutConfig();
loadYoloModels();
loadImageList();






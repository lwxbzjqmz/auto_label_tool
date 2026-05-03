const imageListEl = document.getElementById("imageList");
const imageCountEl = document.getElementById("imageCount");
const currentImageNameEl = document.getElementById("currentImageName");
const statusTextEl = document.getElementById("statusText");
const refreshBtn = document.getElementById("refreshBtn");
const autoDetectBtn = document.getElementById("autoDetectBtn")

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

const boxListEl = document.getElementById("boxList");
const boxCountEl = document.getElementById("boxCount");
const saveXmlBtn = document.getElementById("saveXmlBtn")
const deleteBoxBtn = document.getElementById("deleteBoxBtn");
const clearBoxesBtn = document.getElementById("clearBoxesBtn");

let currentImageName = null;
let currentImage = null;

let boxes = [];
let selectedBoxIndex = -1;

let isDrawing = false;
let startPoint = null;
let currentPoint = null;

// 图片在 canvas 中的显示信息
let imageView = {
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    scale: 1
};


async function loadImageList() {
    statusTextEl.textContent = "状态：正在加载图片列表...";

    try {
        const response = await fetch("/api/images");
        const data = await response.json();

        if (!data.success) {
            statusTextEl.textContent = "状态：加载图片列表失败";
            return;
        }

        renderImageList(data.images);
        imageCountEl.textContent = `图片数量：${data.count}`;
        statusTextEl.textContent = "状态：图片列表加载完成";
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：无法连接后端";
    }
}


function renderImageList(images) {
    imageListEl.innerHTML = "";

    if (images.length === 0) {
        const li = document.createElement("li");
        li.textContent = "暂无图片，请把图片放入 data/JPEGImages";
        imageListEl.appendChild(li);
        return;
    }

    for (const imageName of images) {
        const li = document.createElement("li");
        li.textContent = imageName;

        li.addEventListener("click", () => {
            selectImage(imageName, li);
        });

        imageListEl.appendChild(li);
    }
}


function selectImage(imageName, liElement) {
    currentImageName = imageName;
    boxes = [];
    selectedBoxIndex = -1;

    document.querySelectorAll("#imageList li").forEach(li => {
        li.classList.remove("active");
    });
    liElement.classList.add("active");

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
    ctx.fillStyle = "#777";
    ctx.font = "18px Arial";
    ctx.textAlign = "center";
    ctx.fillText("请从左侧选择一张图片", canvas.width / 2, canvas.height / 2);
}


function drawAllBoxes() {
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const canvasBox = imageBoxToCanvasBox(box);

        ctx.lineWidth = i === selectedBoxIndex ? 3 : 2;
        ctx.strokeStyle = i === selectedBoxIndex ? "#ff0000" : "#00aa00";

        ctx.strokeRect(
            canvasBox.x,
            canvasBox.y,
            canvasBox.w,
            canvasBox.h
        );

        ctx.fillStyle = i === selectedBoxIndex ? "#ff0000" : "#00aa00";
        ctx.font = "14px Arial";
        ctx.fillText(
            `object ${i + 1}`,
            canvasBox.x + 4,
            canvasBox.y + 16
        );
    }
}


function drawTempBox() {
    const x = Math.min(startPoint.x, currentPoint.x);
    const y = Math.min(startPoint.y, currentPoint.y);
    const w = Math.abs(currentPoint.x - startPoint.x);
    const h = Math.abs(currentPoint.y - startPoint.y);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#0066ff";
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


function findBoxAtCanvasPoint(point) {
    for (let i = boxes.length - 1; i >= 0; i--) {
        const b = imageBoxToCanvasBox(boxes[i]);

        if (
            point.x >= b.x &&
            point.x <= b.x + b.w &&
            point.y >= b.y &&
            point.y <= b.y + b.h
        ) {
            return i;
        }
    }

    return -1;
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

        const li = document.createElement("li");
        li.textContent = `${i + 1}: [${box[0]}, ${box[1]}, ${box[2]}, ${box[3]}]`;

        if (i === selectedBoxIndex) {
            li.classList.add("active");
        }

        li.addEventListener("click", () => {
            selectedBoxIndex = i;
            renderBoxList();
            redrawCanvas();
        });

        boxListEl.appendChild(li);
    }
}

async function saveXml() {
    if (!currentImageName) {
        statusTextEl.textContent = "状态：请先选择一张图片";
        return;
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
            return;
        }

        const result = data.result;
        statusTextEl.textContent = `状态：保存成功，${result.xml_name}，框数量 ${result.box_count}`;
    } catch (error) {
        console.error(error);
        statusTextEl.textContent = "状态：保存失败，无法连接后端";
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

async function autoDetect() {
    if (!currentImageName) {
        statusTextEl.textContent = "状态：请先选择一张图片";
        return;
    }

    const payload = {
        image_name: currentImageName
    };

    statusTextEl.textContent = "状态：正在自动标注...";

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

        // 当前策略：自动标注结果直接替换当前 boxes
        // 后面如果你想“追加到已有框”，这里可以改成 boxes = boxes.concat(data.boxes)
        boxes = data.boxes || [];
        selectedBoxIndex = -1;

        renderBoxList();
        redrawCanvas();

        statusTextEl.textContent = `状态：自动标注完成，框数量 ${boxes.length}`;
    } catch (error) {
        console.error("自动标注请求失败：", error);
        statusTextEl.textContent = "状态：自动标注失败，无法连接后端";
    }
}

canvas.addEventListener("mousedown", (event) => {
    if (!currentImage) {
        return;
    }

    resizeCanvasToDisplaySize();
    updateImageView();

    const point = getMousePosition(event);

    if (!isPointInsideDisplayedImage(point)) {
        return;
    }

    const hitIndex = findBoxAtCanvasPoint(point);

    if (hitIndex !== -1) {
        selectedBoxIndex = hitIndex;
        renderBoxList();
        redrawCanvas();
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

    // 限制临时框不要画到图片显示区域外面
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

    // 太小的框不要保存，防止误点
    if (width >= 3 && height >= 3) {
        boxes.push(newBox);
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


deleteBoxBtn.addEventListener("click", () => {
    if (selectedBoxIndex < 0 || selectedBoxIndex >= boxes.length) {
        statusTextEl.textContent = "状态：请先选中一个框";
        return;
    }

    boxes.splice(selectedBoxIndex, 1);
    selectedBoxIndex = -1;

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = "状态：已删除选中框";
});


clearBoxesBtn.addEventListener("click", () => {
    boxes = [];
    selectedBoxIndex = -1;

    renderBoxList();
    redrawCanvas();

    statusTextEl.textContent = "状态：已清空所有框";
});

autoDetectBtn.addEventListener("click", autoDetect);
refreshBtn.addEventListener("click", loadImageList);
saveXmlBtn.addEventListener("click", saveXml);

window.addEventListener("resize", () => {
    resizeCanvasToDisplaySize();
    redrawCanvas();
});


// 页面打开后初始化
resizeCanvasToDisplaySize();
redrawCanvas();
loadImageList();
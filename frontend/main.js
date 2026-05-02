const imageListEl = document.getElementById("imageList");
const imageCountEl = document.getElementById("imageCount");
const previewImageEl = document.getElementById("previewImage");
const currentImageNameEl = document.getElementById("currentImageName");
const statusTextEl = document.getElementById("statusText");
const refreshBtn = document.getElementById("refreshBtn");

let currentImageName = null;


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

    document.querySelectorAll("#imageList li").forEach(li => {
        li.classList.remove("active");
    });
    liElement.classList.add("active");

    previewImageEl.src = `/api/image/${encodeURIComponent(imageName)}`;
    previewImageEl.style.display = "block";

    currentImageNameEl.textContent = `当前图片：${imageName}`;
    statusTextEl.textContent = "状态：图片已加载";
}


refreshBtn.addEventListener("click", loadImageList);


// 页面打开后自动加载图片列表
loadImageList();
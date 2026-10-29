const imageUpload = document.getElementById("imageUpload");
const filterSelect = document.getElementById("filterSelect");
const applyBtn = document.getElementById("applyBtn");
const resetBtn = document.getElementById("resetBtn");

const originalCanvas = document.getElementById("originalCanvas");
const processedCanvas = document.getElementById("processedCanvas");

const originalCtx = originalCanvas.getContext("2d");
const processedCtx = processedCanvas.getContext("2d");

const kernelDisplay = document.getElementById("kernelDisplay");
const filterDescription = document.getElementById("filterDescription");

let originalImageData = null;
let uploadedImage = new Image();

const kernels = {
    mean: [
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9]
    ],

    gaussian: [
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16]
    ],

    sharpen: [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],

    laplacian: [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ],

    sobelX: [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],

    sobelY: [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
};


// Load uploaded image
imageUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (!file) {
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        uploadedImage.onload = function () {
            const maxWidth = 700;
            const maxHeight = 500;

            let width = uploadedImage.width;
            let height = uploadedImage.height;

            // Resize large images while maintaining aspect ratio
            if (width > maxWidth || height > maxHeight) {
                const widthRatio = maxWidth / width;
                const heightRatio = maxHeight / height;
                const ratio = Math.min(widthRatio, heightRatio);

                width = Math.floor(width * ratio);
                height = Math.floor(height * ratio);
            }

            originalCanvas.width = width;
            originalCanvas.height = height;

            processedCanvas.width = width;
            processedCanvas.height = height;

            originalCtx.clearRect(
                0,
                0,
                originalCanvas.width,
                originalCanvas.height
            );

            originalCtx.drawImage(
                uploadedImage,
                0,
                0,
                width,
                height
            );

            originalImageData = originalCtx.getImageData(
                0,
                0,
                width,
                height
            );

            processedCtx.putImageData(originalImageData, 0, 0);

            filterSelect.value = "original";
            showFilterInformation("original");
        };

        uploadedImage.src = e.target.result;
    };

    reader.readAsDataURL(file);
});


// Apply selected filter
applyBtn.addEventListener("click", function () {
    if (!originalImageData) {
        alert("Please upload an image first.");
        return;
    }

    const selectedFilter = filterSelect.value;

    let processedImageData;

    if (selectedFilter === "original") {
        processedImageData = originalImageData;
    }

    else if (selectedFilter === "mean") {
        processedImageData = applyColorConvolution(
            originalImageData,
            kernels.mean
        );
    }

    else if (selectedFilter === "gaussian") {
        processedImageData = applyColorConvolution(
            originalImageData,
            kernels.gaussian
        );
    }

    else if (selectedFilter === "sharpen") {
        processedImageData = applyColorConvolution(
            originalImageData,
            kernels.sharpen
        );
    }

    else if (selectedFilter === "sobel") {
        processedImageData = applySobelFilter(originalImageData);
    }

    else if (selectedFilter === "laplacian") {
        processedImageData = applyLaplacianFilter(originalImageData);
    }

    processedCtx.putImageData(processedImageData, 0, 0);

    showFilterInformation(selectedFilter);
});


// Reset to original image
resetBtn.addEventListener("click", function () {
    if (!originalImageData) {
        alert("Please upload an image first.");
        return;
    }

    processedCtx.putImageData(originalImageData, 0, 0);

    filterSelect.value = "original";
    showFilterInformation("original");
});


// Color-preserving 2D FIR convolution
function applyColorConvolution(imageData, kernel) {
    const width = imageData.width;
    const height = imageData.height;

    const source = imageData.data;
    const output = new ImageData(width, height);
    const destination = output.data;

    const kernelSize = kernel.length;
    const offset = Math.floor(kernelSize / 2);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            let red = 0;
            let green = 0;
            let blue = 0;

            for (let ky = 0; ky < kernelSize; ky++) {
                for (let kx = 0; kx < kernelSize; kx++) {

                    let pixelX = x + kx - offset;
                    let pixelY = y + ky - offset;

                    // Border handling
                    pixelX = Math.max(0, Math.min(width - 1, pixelX));
                    pixelY = Math.max(0, Math.min(height - 1, pixelY));

                    const sourceIndex =
                        (pixelY * width + pixelX) * 4;

                    const weight = kernel[ky][kx];

                    red += source[sourceIndex] * weight;
                    green += source[sourceIndex + 1] * weight;
                    blue += source[sourceIndex + 2] * weight;
                }
            }

            const outputIndex = (y * width + x) * 4;

            destination[outputIndex] = clamp(red);
            destination[outputIndex + 1] = clamp(green);
            destination[outputIndex + 2] = clamp(blue);
            destination[outputIndex + 3] = 255;
        }
    }

    return output;
}


// Sobel edge detection
function applySobelFilter(imageData) {
    const width = imageData.width;
    const height = imageData.height;

    const source = imageData.data;
    const output = new ImageData(width, height);
    const destination = output.data;

    const gx = kernels.sobelX;
    const gy = kernels.sobelY;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            let gradientX = 0;
            let gradientY = 0;

            for (let ky = 0; ky < 3; ky++) {
                for (let kx = 0; kx < 3; kx++) {

                    let pixelX = x + kx - 1;
                    let pixelY = y + ky - 1;

                    pixelX = Math.max(0, Math.min(width - 1, pixelX));
                    pixelY = Math.max(0, Math.min(height - 1, pixelY));

                    const index = (pixelY * width + pixelX) * 4;

                    // Convert pixel to grayscale intensity
                    const gray =
                        0.299 * source[index] +
                        0.587 * source[index + 1] +
                        0.114 * source[index + 2];

                    gradientX += gray * gx[ky][kx];
                    gradientY += gray * gy[ky][kx];
                }
            }

            const magnitude = Math.sqrt(
                gradientX * gradientX +
                gradientY * gradientY
            );

            const edgeValue = clamp(magnitude);

            const outputIndex = (y * width + x) * 4;

            destination[outputIndex] = edgeValue;
            destination[outputIndex + 1] = edgeValue;
            destination[outputIndex + 2] = edgeValue;
            destination[outputIndex + 3] = 255;
        }
    }

    return output;
}


// Laplacian edge detection
function applyLaplacianFilter(imageData) {
    const width = imageData.width;
    const height = imageData.height;

    const source = imageData.data;
    const output = new ImageData(width, height);
    const destination = output.data;

    const kernel = kernels.laplacian;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            let value = 0;

            for (let ky = 0; ky < 3; ky++) {
                for (let kx = 0; kx < 3; kx++) {

                    let pixelX = x + kx - 1;
                    let pixelY = y + ky - 1;

                    pixelX = Math.max(0, Math.min(width - 1, pixelX));
                    pixelY = Math.max(0, Math.min(height - 1, pixelY));

                    const index = (pixelY * width + pixelX) * 4;

                    const gray =
                        0.299 * source[index] +
                        0.587 * source[index + 1] +
                        0.114 * source[index + 2];

                    value += gray * kernel[ky][kx];
                }
            }

            const edgeValue = clamp(Math.abs(value));

            const outputIndex = (y * width + x) * 4;

            destination[outputIndex] = edgeValue;
            destination[outputIndex + 1] = edgeValue;
            destination[outputIndex + 2] = edgeValue;
            destination[outputIndex + 3] = 255;
        }
    }

    return output;
}


// Keep pixel values between 0 and 255
function clamp(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}


// Display filter information
function showFilterInformation(filter) {
    const descriptions = {
        original: "Displays the original uploaded image.",

        mean: "Mean filter reduces noise by averaging neighbouring pixels. It may produce slight blur.",

        gaussian: "Gaussian filter smooths the image and reduces noise while preserving details better than the mean filter.",

        sharpen: "Sharpening filter enhances edges and makes a slightly blurred image appear clearer.",

        sobel: "Sobel filter detects object boundaries and edges. The output is displayed in grayscale.",

        laplacian: "Laplacian filter detects rapid intensity changes and fine image details. The output is displayed in grayscale."
    };

    const selectedKernel = kernels[filter];

    if (filter === "original") {
        kernelDisplay.textContent = "No kernel selected";
    }

    else if (filter === "sobel") {
        kernelDisplay.textContent =
            "Sobel X:\n" +
            formatKernel(kernels.sobelX) +
            "\n\nSobel Y:\n" +
            formatKernel(kernels.sobelY);
    }

    else if (selectedKernel) {
        kernelDisplay.textContent = formatKernel(selectedKernel);
    }

    filterDescription.textContent =
        descriptions[filter] || "Select a filter to view its description.";
}


// Format kernel for display
function formatKernel(kernel) {
    return kernel
        .map(row => row.map(value => value.toFixed(3)).join("   "))
        .join("\n");
}
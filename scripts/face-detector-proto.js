function FaceDetector() {
    this.eyesRegionRect = null;
}

FaceDetector.prototype.detectFace = function () {
    faceapi.matchDimensions.call(this, faceLandmarksCanvas, webcamElement);
    setInterval(async () => {
        const detection = await faceapi.detectAllFaces.call(this, webcamElement, new faceapi.TinyFaceDetectorOptions({
            scoreThreshold: 0.4,
            inputSize: 224
        })).withFaceLandmarks();
        const resizedDetection = faceapi.resizeResults(detection,  webcamElement);
        faceLandmarksCanvasContext.clearRect(
            0,
            0,
            faceLandmarksCanvas.width,
            faceLandmarksCanvas.height
        );
        drawFaceLandmarksCheckbox && faceapi.draw.drawFaceLandmarks(faceLandmarksCanvasContext, resizedDetection);

        if (resizedDetection.length) {
            const eyesRegionRect = this.detectEyesRegion(
                resizedDetection[0].landmarks.getLeftEye(),
                resizedDetection[0].landmarks.getRightEye()
            );
            this.eyesRegionRect = eyesRegionRect;
            if (highlightEyesRegion) {
                faceLandmarksCanvasContext.strokeStyle = "#00a8ff";
                faceLandmarksCanvasContext.strokeRect(eyesRegionRect[0], eyesRegionRect[1], eyesRegionRect[2], eyesRegionRect[3]);
            }
            const resizeFactorX = webcamElement.videoWidth / webcamElement.width;
            const resizeFactorY = webcamElement.videoHeight / webcamElement.height;

            eyesRegionCanvasContext.drawImage(
                webcamElement,
                eyesRegionRect[0] * resizeFactorX, eyesRegionRect[1] * resizeFactorY,
                eyesRegionRect[2] * resizeFactorX, eyesRegionRect[3] * resizeFactorY,
                0, 0, eyesRegionCanvas.width, eyesRegionCanvas.height
            );
        }
    }, 100);
};

FaceDetector.prototype.detectEyesRegion = function (leftEye, rightEye) {
    const minX = leftEye[0].x - 10;
    const maxX = rightEye[3].x + 10;
    const minY = Math.min(
        leftEye[1].y,
        leftEye[2].y,
        rightEye[1].y,
        rightEye[2].y,
    ) - 10;
    const maxY = Math.max(
        leftEye[4].y,
        leftEye[5].y,
        rightEye[4].y,
        rightEye[5].y,
    ) + 6;

    const width = maxX - minX;
    const height = maxY - minY;

    return [minX, minY, width, height];
};

FaceDetector.prototype.captureEyesRegionImage = function() {
    return tf.tidy(() => {
        return tf.browser.fromPixels(eyesRegionCanvas)
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(127))
            .sub(tf.scalar(1));
    });
};

FaceDetector.prototype.getEyesRegionMetaInfo = function(mirror) {
    let x = this.eyesRegionRect[0] + this.eyesRegionRect[2] / 2;
    let y = this.eyesRegionRect[1] + this.eyesRegionRect[3] / 2;
    x = (x / webcamElement.width) * 2 - 1;
    y = (y / webcamElement.height) * 2 - 1;
    const rectWidth = this.eyesRegionRect[2] / webcamElement.width;
    const rectHeight = this.eyesRegionRect[3] / webcamElement.height;

    if (mirror) {
        x = 1 - x;
        y = 1 - y;
    }

    return tf.tidy(
        () => tf.tensor1d([x, y, rectWidth, rectHeight]).expandDims(0)
    )
};

FaceDetector.prototype.captureExample = function(dataset, mouse) {
    tf.tidy.call(this, () => {
        const img = this.captureEyesRegionImage();
        const mousePos = mouse.getPosition();
        const metaInfos = tf.keep.call(this, this.getEyesRegionMetaInfo());
        dataset.addExample(img, metaInfos, mousePos);
    });
};

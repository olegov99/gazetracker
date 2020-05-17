$(document).ready(() => {
    // Global Variables
    window.webcamElement = document.getElementById("webcam");
    window.faceLandmarksCanvas = document.getElementById("face-landmarks");
    window.faceLandmarksCanvasContext = faceLandmarksCanvas.getContext("2d");
    window.eyesRegionCanvas = document.getElementById("eyes-region");
    window.eyesRegionCanvasContext = eyesRegionCanvas.getContext("2d");

    window.webcam = new Webcam();
    window.faceDetector = new FaceDetector();
    window.mouseProto = new Mouse();
    window.dataset = new Dataset(eyesRegionCanvas);
    window.eyeGazeDetector = new EyeGazeDetector();

    document.onmousemove = mouseProto.onMove.bind(mouseProto);

    // eyeGazeDetector.model = await tf.loadLayersModel('./egd-model/model.json');

    Promise.all([
        // faceapi.loadSsdMobilenetv1Model('./'),
        // faceapi.loadTinyFaceDetectorModel('./'),
        // faceapi.loadFaceLandmarkTinyModel('./'),
        // faceapi.loadFaceExpressionModel('./'),
        // faceapi.loadFaceDetectionModel('./'),
        // faceapi.loadFaceRecognitionModel('./'),
        // faceapi.loadFaceLandmarkModel('./')
        faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
        faceapi.nets.faceExpressionNet.loadFromUri("./models"),
    ]).then(webcam.startStream);

    webcamElement.addEventListener("play", () => {
        faceDetector.detectFace();
    });
});

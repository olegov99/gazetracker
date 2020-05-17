function Webcam() {
}

Webcam.prototype.onStream = (stream) => {
    if ("srcObject" in webcam) {
        webcam.srcObject = stream;
    } else {
        webcam.src = window.URL && window.URL.createObjectURL(stream);
    }
};

Webcam.prototype.startStream = () => {
    if (navigator.getUserMedia) {
        navigator.getUserMedia(
            {video: true},
            (stream) => {
                    webcamElement.srcObject = stream;
            },
            error => console.error(error)
        );
    }
};

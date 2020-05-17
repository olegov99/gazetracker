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

    if (navigator.mediaDevices) {
        navigator.mediaDevices.getUserMedia({
            video: true,
        }).then((stream) => {
            webcamElement.srcObject = stream;
        }).catch((err) => {
            console.error(err);
        });
    } else if (navigator.getUserMedia) {
        navigator.getUserMedia(
            {video: true},
            (stream) => {
                    webcamElement.srcObject = stream;
            },
            error => console.error(error)
        );
    }
};

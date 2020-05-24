$(document).ready(() => {

    tfvis.visor().close();

    $('#dots-grid').click(function(e) {
        faceDetector.captureExample(dataset, mouseProto);
        e.preventDefault();
    });

    // Creating dots grid for collecting dataset
    for (let i = 1; i <= 50; i++) {
        $('<div class="box"><div class="dot"></div></div>').appendTo('#dots-grid');
    }

    $('#train').click(function(e) {
        eyeGazeDetector.train(dataset);
    });

    const $target = $("#target");
    const targetSize = $target.outerWidth();

    function moveTarget() {
        if (eyeGazeDetector.model == null) {
            return;
        }

        faceDetector.eyesRegionRect && eyeGazeDetector.predict(faceDetector).then(prediction => {
            const left = prediction[0] * ($('body').width() - targetSize);
            const top = prediction[1] * ($('body').height() - targetSize);

            if (drawHeatmapCheckbox) {
                const dataPoint = {
                    x: left,
                    y: top,
                    value: 100
                };

                heatmapInstance.addData(dataPoint);
            }

            $target.css('left', left + 'px');
            $target.css('top', top + 'px');
        });
    }

    setInterval(moveTarget, 100);

    $("#extend-btn").on("click", () => {
        $(".egd-estimation__sidebar").toggleClass("_opened");
    });


    // Face Landmarks Checkbox
    $faceLandmarksCheckbox = $('#face-landmarks-checkbox');
    window.drawFaceLandmarksCheckbox = $faceLandmarksCheckbox.checked;
    $faceLandmarksCheckbox.change(function() {
        window.drawFaceLandmarksCheckbox = this.checked;
    });

    // Highlight Eyes Region
    $eyesRegionCheckbox = $('#eyes-region-checkbox');
    window.highlightEyesRegion = $eyesRegionCheckbox.checked;
    $eyesRegionCheckbox.change(function() {
        window.highlightEyesRegion = this.checked;
    });

    // Dataset collecting dots checkbox
    $datasetDotsCheckbox = $('#dots-grid-checkbox');
    !$datasetDotsCheckbox.checked && $('#dots-grid').css("opacity", "0");
    $datasetDotsCheckbox.change(function() {
        if (this.checked) {
            $('#dots-grid').css("opacity", "1");
        } else {
            $('#dots-grid').css("opacity", "0");
        }
    });

    // Draw heatmap
    $drawHeatmpCheckbox = $('#show-heatmap-checkbox');
    window.drawHeatmapCheckbox = $drawHeatmpCheckbox.checked;
    !drawHeatmapCheckbox && $('#heatmap').css("display", "hidden");
    $drawHeatmpCheckbox.change(function() {
        window.drawHeatmapCheckbox = this.checked;
        if (this.checked) {
            $('#target').css("opacity", "0");
        }
        if(!this.checked) {
            $('#heatmap').css("display", "hidden");
            $('#target').css("opacity", "1");
            heatmapInstance.repaint();
        }
    });

    $('#train-btn').click(function(e) {
        eyeGazeDetector.train(dataset);
    });

    window.infoBoxUpdate = function (train, val) {
        $("#train-num").html(train);
        $("#val-num").html(val);
    };

    $("#visualize-btn").on("click", () => {
        if (!tfvis.visor().isOpen()) {
            tfvis.visor().toggle();
        }
    });

    $("#save-model-btn").on("click", async () => {
        await eyeGazeDetector.model.save('downloads://model');
    });

    $("#reset-model-btn").on("click", async () => {
        eyeGazeDetector.model = null;
    });

    $('#save-dataset-btn').click(function(e) {
        const data = dataset.saveDataset();
        const json = JSON.stringify(data);
        download(json, 'dataset.json', 'text/plain');
    });

    function download(content, fileName, contentType) {
        const a = document.createElement('a');
        const file = new Blob([content], {
            type: contentType,
        });
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
    }

    $("#upload-dataset-btn").click(function() {
        $("#dataset-uploader").trigger("click");
    });

    $('#dataset-uploader').change(function(e) {
        for (let i = 0; i < e.target.files.length; i++) {
            const file = e.target.files[i];
            const reader = new FileReader();

            reader.onload = function() {
                const data = reader.result;
                const json = JSON.parse(data);
                dataset.uploadDataset(json);
            };

            reader.readAsBinaryString(file);
        }
    });

    const config = {
        container: document.getElementById('heatmap'),
        radius: 40,
        maxOpacity: .5,
        minOpacity: 0,
        blur: .85
    };

    window.heatmapInstance = h337.create(config);

});

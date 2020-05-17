function EyeGazeDetector() {
    this.model = null;
    this.epochsTrained = 0;
    this.isTraining = false;
}

EyeGazeDetector.prototype.createModel = function (dataset) {
    const inputImage = tf.input({
        name: "image",
        shape: [dataset.imageHeight, dataset.imageWidth, 3]
    });
    const inputMetaInfo = tf.input({
        name: "metaInfo",
        shape: [4]
    });

    const convFirst = tf.layers.conv2d({
        kernelSize: 5,
        filters: 32,
        strides: 1,
        activation: "relu",
        kernelInitializer:"glorotUniform"
    }).apply(inputImage);

    const maxPoolingFirst = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }).apply(convFirst);

    const dropoutFirst = tf.layers.dropout(0.5).apply(maxPoolingFirst);

    const convSecond = tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: "relu",
        kernelInitializer:"glorotUniform"
    }).apply(dropoutFirst);

    const maxPoolingSecond = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }).apply(convSecond);

    const dropoutSecond = tf.layers.dropout(0.5).apply(maxPoolingSecond);

    const convThird= tf.layers.conv2d({
        kernelSize: 1,
        filters: 32,
        strides: 1,
        activation: "relu",
        kernelInitializer:"glorotUniform"
    }).apply(dropoutSecond);

    const maxPoolingThird = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }).apply(convThird);

    const flatten = tf.layers.flatten().apply(maxPoolingThird);

    const dropoutThird = tf.layers.dropout(0.5).apply(flatten);

    const concat = tf.layers.concatenate().apply([dropoutThird, inputMetaInfo]);

    const output = tf.layers.dense({
        units: 2,
        activation: 'tanh',
        kernelInitializer: 'glorotUniform',
    }).apply(concat);

    return tf.model({
        inputs: [inputImage, inputMetaInfo],
        outputs: output,
    });
};

EyeGazeDetector.prototype.train = async function(dataset) {
    this.isTraining = true;
    const epochs = 15;

    let batchSize = Math.floor(dataset.train.num * 0.1);
    batchSize = Math.max(2, Math.min(batchSize, 64));

    if (this.model == null) {
        this.model = this.createModel(dataset);
    }

    let bestEpoch = -1;
    let bestTrainLoss = Number.MAX_SAFE_INTEGER;
    let bestValLoss = Number.MAX_SAFE_INTEGER;

    this.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['accuracy']
    });

    const history = [];

    await this.model.fit(dataset.train.x, dataset.train.y, {
        batchSize: batchSize,
        epochs: epochs,
        shuffle: true,
        validationData: [dataset.val.x, dataset.val.y],
        callbacks: {
            onEpochEnd: async function (epoch, logs) {
                console.info('Epoch', epoch, 'losses:', logs);
                this.epochsTrained += 1;

                if (logs.val_loss < bestValLoss) {
                    bestEpoch = epoch;
                    bestTrainLoss = logs.loss;
                    bestValLoss = logs.val_loss;

                    // await model.cnn.save(bestModelPath);
                }

                history.push(logs);
                await tfvis.show.history({
                    name: "train",
                    tab: "train",
                    styles: {height: '600px'}
                }, history, ['loss', 'acc', 'val_loss', 'val_acc']);

                return await tf.nextFrame();
            },
            onTrainEnd: async function () {

                this.epochsTrained -= epochs - bestEpoch;

                // model.cnn = await tf.loadLayersModel(bestModelPath);

                this.isTraining = false;
            },
        },
    });
};

EyeGazeDetector.prototype.predict = async function(faceDetector) {
    const rawImg = faceDetector.captureEyesRegionImage();
    const img = await convertImage(rawImg);
    const metaInfos = faceDetector.getEyesRegionMetaInfo();
    const prediction = this.model.predict([img, metaInfos]);
    const predictionData = await prediction.data();

    tf.dispose([img, metaInfos, prediction]);

    return [predictionData[0] + 0.5, predictionData[1] + 0.5];
};

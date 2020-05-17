const TRAIN_SET = "train";
const VAL_SET = "val";

function Dataset(eyesRegionCanvas) {
    this.imageWidth = eyesRegionCanvas.width;
    this.imageHeight = eyesRegionCanvas.height;
    this.train = {
        num: 0,
        x: null,
        y: null
    };
    this.val = {
        num: 0,
        x: null,
        y: null
    }
}

Dataset.prototype.selectDataset = function() {
    if (this.train.num == 0) {
        return TRAIN_SET;
    }

    if (this.val.num == 0) {
        return VAL_SET;
    }

    return Math.random() < 0.2 ? VAL_SET : TRAIN_SET;
};

Dataset.prototype.addToDataset = function(image, metaInfos, target, key) {
    const set = this[key];

    if (set.x == null) {
        set.x = [tf.keep(image), tf.keep(metaInfos)];
        set.y = tf.keep(target);
    } else {
        const oldImage = set.x[0];
        set.x[0] = tf.keep(oldImage.concat(image, 0));

        const oldEyePos = set.x[1];
        set.x[1] = tf.keep(oldEyePos.concat(metaInfos, 0));

        const oldY = set.y;
        set.y = tf.keep(oldY.concat(target, 0));

        tf.dispose([oldImage, oldEyePos, oldY, target]);
    }

    set.num += 1;
};

Dataset.prototype.addExample = async function(image, metaInfos, target, dontDispose) {
    target[0] = target[0] - 0.5;
    target[1] = target[1] - 0.5;
    target = tf.keep(
        tf.tidy(function() {
            return tf.tensor1d(target).expandDims(0);
        }),
    );
    const key = this.selectDataset();
    const convertedImage = await convertImage(image);
    this.addToDataset(convertedImage, metaInfos, target, key);
    infoBoxUpdate(this.train.num, this.val.num);
    if (!dontDispose) {
        tf.dispose(image, metaInfos);
    }
};

Dataset.prototype.saveDataset = function() {
    const tensorToArray = function(t) {
        const typedArray = t.dataSync();
        return Array.prototype.slice.call(typedArray);
    };

    return {
        inputWidth: this.imageWidth,
        inputHeight: this.imageHeight,
        train: {
            shapes: {
                x0: this.train.x[0].shape,
                x1: this.train.x[1].shape,
                y: this.train.y.shape,
            },
            num: this.train.num,
            x: this.train.x && [
                tensorToArray(this.train.x[0]),
                tensorToArray(this.train.x[1]),
            ],
            y: tensorToArray(this.train.y),
        },
        val: {
            shapes: {
                x0: this.val.x[0].shape,
                x1: this.val.x[1].shape,
                y: this.val.y.shape,
            },
            num: this.val.num,
            x: this.val.x && [
                tensorToArray(this.val.x[0]),
                tensorToArray(this.val.x[1]),
            ],
            y: tensorToArray(this.val.y),
        },
    };
};

Dataset.prototype.uploadDataset = function(data) {
    this.imageWidth = data.inputWidth;
    this.imageHeight = data.inputHeight;
    this.train.num += data.train.num;
    if (this.train.x === null) {
        this.train.x = data.train.x && [
            tf.tensor(data.train.x[0], data.train.shapes.x0),
            tf.tensor(data.train.x[1], data.train.shapes.x1),
        ];
        this.train.y = tf.tensor(data.train.y, data.train.shapes.y);
    } else {
        const oldImages = this.train.x[0];
        const newImages = tf.tensor(data.train.x[0], data.train.shapes.x0);
        this.train.x[0] = tf.keep(oldImages.concat(newImages, 0));

        const oldEyePos = this.train.x[1];
        const newEyePos = tf.tensor(data.train.x[1], data.train.shapes.x1);
        this.train.x[1] = tf.keep(oldEyePos.concat(newEyePos, 0));

        const oldY = this.train.y;
        const newY = tf.tensor(data.train.y, data.train.shapes.y);
        this.train.y = tf.keep(oldY.concat(newY, 0));

        tf.dispose([oldImages, oldEyePos, oldY, newImages, newEyePos, newY]);
    }
    this.val.num += data.val.num;
    if (this.val.x === null) {
        this.val.x = data.val.x && [
            tf.tensor(data.val.x[0], data.val.shapes.x0),
            tf.tensor(data.val.x[1], data.val.shapes.x1),
        ];
        this.val.y = tf.tensor(data.val.y, data.val.shapes.y);
    } else {
        const oldImages = this.val.x[0];
        const newImages = tf.tensor(data.val.x[0], data.val.shapes.x0);
        this.val.x[0] = tf.keep(oldImages.concat(newImages, 0));

        const oldEyePos = this.val.x[1];
        const newEyePos = tf.tensor(data.val.x[1], data.val.shapes.x1);
        this.val.x[1] = tf.keep(oldEyePos.concat(newEyePos, 0));

        const oldY = this.val.y;
        const newY = tf.tensor(data.val.y, data.val.shapes.y);
        this.val.y = tf.keep(oldY.concat(newY, 0));

        tf.dispose([oldImages, oldEyePos, oldY, newImages, newEyePos, newY]);
    }
    infoBoxUpdate(this.train.num, this.val.num);
};

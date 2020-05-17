function rgbToGrayscale(imgArray, n, x, y) {
    let r = (imgArray[n][x][y][0] + 1) / 2;
    let g = (imgArray[n][x][y][1] + 1) / 2;
    let b = (imgArray[n][x][y][2] + 1) / 2;

    const exponent = 1 / 2.2;
    r = Math.pow(r, exponent);
    g = Math.pow(g, exponent);
    b = Math.pow(b, exponent);

    const gleam = (r + g + b) / 3;
    return gleam * 2 - 1;
}

async function convertImage(image) {
    const imageShape = image.shape;
    const imageArray = await image.array();
    const w = imageShape[1];
    const h = imageShape[2];

    const data = [new Array(w)];
    const promises = [];
    for (let x = 0; x < w; x++) {
        data[0][x] = new Array(h);

        for (let y = 0; y < h; y++) {
            const grayValue = rgbToGrayscale(imageArray, 0, x, y);
            data[0][x][y] = [grayValue, (x / w) * 2 - 1, (y / h) * 2 - 1];
        }
    }

    await Promise.all(promises);
    return tf.tensor(data);
}

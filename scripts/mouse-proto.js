function Mouse() {
    this.position = {
        x: 0,
        y: 0
    };
}

Mouse.prototype.onMove = function(event) {
    this.position.x = event.clientX / $('body').width();
    this.position.y = event.clientY / $('body').height();
};

Mouse.prototype.getPosition = function() {
  return [this.position.x, this.position.y];
};

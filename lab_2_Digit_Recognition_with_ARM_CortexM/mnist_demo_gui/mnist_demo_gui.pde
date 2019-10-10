import processing.serial.Serial;

Serial _port;

@Override
void setup()
{
    size(640, 280, P2D);
    background(0xFF);

    textSize(144); // Workaround P2D
    textAlign(CENTER);

    strokeWeight(20);
    noStroke();

    // Print all avaiable Serial ports.
    printArray(Serial.list());
    // According to the Serial port list, choose the one with a connection to the board.
    int portNo = 0;
    _port = new Serial(this, Serial.list()[portNo], 115200);
    
    _clear();
    _redraw(Button.NONE);
}

@Override
void draw() {}

@Override
void mouseDragged()
{
    if (mouseX < 280) {
        stroke(0xFF);
        line(pmouseX, pmouseY, mouseX, mouseY);
        noStroke();
    }
}

@Override
void mouseMoved()
{
    _redraw(_button());
}

@Override
void mouseClicked()
{
    switch (_button()) {
        case SUBMIT:
            _say(_recognize());
            break;
        case CLEAR:
            _clear();
            break;
    }
}

@Override
void keyPressed()
{
    switch (keyCode) {
        case ENTER:
        case RETURN:
        case ' ':
        case 'S':
            _say(_recognize());
            break;
        case BACKSPACE:
        case DELETE:
        case 'C':
            _clear();
            break;
    }
}

private void _say(int number)
{
    fill(0xFF);
    rect(400, 0, 100, 160);
    fill(0);
    textSize(144);

    if (number >= 0) {
        text(number, 460, 140);
    }
    else {
        text('-', 460, 140);
    }
}

boolean result_returned;

void serialEvent(Serial p) {
  result_returned = true;
}

private int _recognize()
{
    final PImage img = get(0, 0, 280, 280);
    img.resize(28, 28);
    img.loadPixels();

    final int size = img.pixels.length;
    final byte data[] = new byte[size];

    for (int i = 0; i < size; ++i)
        data[i] = (byte)(img.pixels[i] >> 1 & 0x7F);

    result_returned = false;
    _port.write(data);
    while (!result_returned) { delay(1); }
    return _port.read();
}

private void _clear()
{
    fill(0);
    rect(0, 0, 280, 280);
}

private void _redraw(Button hover)
{
    fill(hover == Button.SUBMIT ? #007ACC : 0);
    rect(310, 190, 140, 50, 5);

    fill(hover == Button.CLEAR ? #007ACC : 0);
    rect(470, 190, 140, 50, 5);

    fill(0xFF);
    textSize(32);
    text("Submit", 380, 225);
    text("Clear",  540, 225);
}

private Button _button()
{
    if (mouseY >= 190 && mouseY < 240) {
        if (mouseX >= 310 && mouseX < 450)
            return Button.SUBMIT;
        else if (mouseX >= 470 && mouseX < 610)
            return Button.CLEAR;
    }
    return Button.NONE;
}

private enum Button
{
    NONE,
    SUBMIT,
    CLEAR,
}
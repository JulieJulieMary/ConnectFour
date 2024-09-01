from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import io
import subprocess
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://JulieJulieMary.github.io"}})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Open the image file using PIL
    img = Image.open(file.stream)

    # Perform your processing here (this is just an example)
    img.save('image.png')

    subprocess.run(['python', 'minimized.py'])
    with open('move.txt', 'r') as f:
        firstMove = f.readline().strip()
        game = f.read().strip()

    game_array = [list(map(int, row.split())) for row in game.splitlines()]

    # Return the firstMove and game_array as JSON
    response = jsonify({"firstMove": int(firstMove), "game": game_array})
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins for now, restrict in production
    return response
    # Save processed image to a byte buffer
    # buf = io.BytesIO()
    # img.save(buf, format='PNG')
    # buf.seek(0)

    # # Send the processed image back to the client
    # return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

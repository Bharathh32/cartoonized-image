import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import os

# Define the Cartoonizer class
class Cartoonizer:
    def __init__(self, line_size=7, blur_value=7, k=9):
        self.line_size = line_size
        self.blur_value = blur_value
        self.k = k

    def edge_mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, self.blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, self.line_size, self.blur_value)
        return edges

    def color_quantization(self, img):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    def cartoonize(self, img):
        edges = self.edge_mask(img)
        img = self.color_quantization(img)
        blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
        return cartoon

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

cartoonizer = Cartoonizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route to upload file
@app.route('/')
def upload_file():
    return render_template('index.html')

# Route to handle file upload and processing
@app.route('/cartoonize', methods=['POST'])
def cartoonize_file():
    try:
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            # Read the file in memory instead of saving it to disk
            file_stream = BytesIO(file.read())
            img = np.array(Image.open(file_stream).convert('RGB'))
            
            cartoon_img = cartoonizer.cartoonize(img)

            # Convert the cartoon image to a format that can be sent over the web
            pil_img = Image.fromarray(cartoon_img)
            img_io = BytesIO()
            pil_img.save(img_io, 'PNG')
            img_io.seek(0)

            print("Cartoonized image generated successfully.")
            return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print("Error processing image:", e)
        return f"Error processing image: {e}", 500

# Start the Flask application
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

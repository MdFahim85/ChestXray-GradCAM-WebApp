from flask import Flask, request, render_template
import os
from model.gradcam_utils import save_gradcam_image, model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Generate Grad-CAM image and predictions
            gradcam_path, predictions = save_gradcam_image(
                model=model,
                img_path=image_path,
                output_dir=RESULT_FOLDER,
                layer_name='conv2d_4'  # change if your model's last conv layer is different
            )

            return render_template(
                'index.html',
                predictions=predictions,
                original_image=image_path,
                gradcam_image=gradcam_path
            )

        else:
            return render_template('index.html', error="Please upload a valid image file (.png, .jpg, .jpeg)")

    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

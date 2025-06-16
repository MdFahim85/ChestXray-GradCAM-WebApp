import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Disease labels
disease_labels = [
    'Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema',
    'Effusion', 'Atelectasis', 'Pleural_Thickening', 'Pneumothorax', 'Mass',
    'Fibrosis', 'Consolidation', 'Edema', 'Pneumonia', 'No Finding'
]

# Load model once (Flask can load externally too)
model = load_model('model/ChestAidNet.h5')

def get_gradcam_heatmap(model, img_array, layer_name, pred_index=None):
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), preds


def save_gradcam_image(model, img_path, output_dir='static/results', layer_name='conv2d_4', alpha=0.5):
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess image
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Grad-CAM
    heatmap, preds = get_gradcam_heatmap(model, img_array, layer_name)
    top_pred_idx = np.argmax(preds[0])
    top_pred_class = disease_labels[top_pred_idx]
    top_pred_prob = preds[0][top_pred_idx]

    # Resize and apply heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_array = img_to_array(img)

    # Superimpose heatmap
    superimposed_img = heatmap * alpha + original_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    superimposed_pil = Image.fromarray(superimposed_img)

    # Save image
    filename = os.path.basename(img_path).split('.')[0] + '_gradcam.png'
    save_path = os.path.join(output_dir, filename)
    superimposed_pil.save(save_path)

    # Get top 3 predictions
    top_3 = []
    sorted_indices = np.argsort(preds[0])[::-1][:3]
    for idx in sorted_indices:
        top_3.append((disease_labels[idx], float(preds[0][idx])))

    return save_path, top_3


import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
import cv2
import imageio
import time
from flask import Flask, render_template, request, send_from_directory
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tensorflow.keras.models import load_model



SEQ_LENGTH = 20  # Total sequence length
INPUT_LENGTH = 10  # Input frames for the model
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNEL = 1


app = Flask(__name__)

# Define folders for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Updated Model Definition using tf.keras
inp = tf.keras.layers.Input(shape=(None, 64, 64, 1))  # None allows variable time steps

# Construct the ConvLSTM2D layers and Conv3D output layer
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = tf.keras.layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

# Build the model
model = tf.keras.models.Model(inp, x)
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
)






# Define the model path
# Build the model
input_shape = (INPUT_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
output_channels = IMG_CHANNEL
hidden_size = 64

# Load the model with the updated custom objects

print("Model loaded built!")

model_weights_path = 'C:\\Users\\aalia\\Downloads\\lstm\\rnn\\rnn\\predrnn_weights.weights.h5'
model.load_weights(model_weights_path)

print("Model weights loaded successfully!")




def calculate_ssim(original_frames, predicted_frames):
    """
    Calculate the average SSIM between original and predicted frames.

    Args:
        original_frames (list): List of original frames (grayscale).
        predicted_frames (list): List of predicted frames (grayscale).
    
    Returns:
        float: Average SSIM value.
    """
    ssim_values = []
    predicted_frames = [frame / 255.0 for frame in predicted_frames]
    original_frames = [frame.astype(np.float32) for frame in original_frames]
    predicted_frames = [frame.astype(np.float32) for frame in predicted_frames]
    for orig, pred in zip(original_frames, predicted_frames):
        print(f"Original frame range: {orig.min()} to {orig.max()}")
        print(f"Predicted frame range: {pred.min()} to {pred.max()}")
        # Specify data_range=1.0 since frames are normalized to [0, 1]
        score, _ = ssim(orig, pred, full=True, data_range=1.0)
        ssim_values.append(score)
    return np.mean(ssim_values)


def calculate_mse(original_frames, predicted_frames):
    """
    Calculate the average MSE between original and predicted frames.

    Args:
        original_frames (list): List of original frames (grayscale).
        predicted_frames (list): List of predicted frames (grayscale).
    
    Returns:
        float: Average MSE value.
    """
    # Ensure both lists have the same number of frames
    assert len(original_frames) == len(predicted_frames), "Mismatch in frame counts"

    mse_values = []

    # Normalize frames to [0, 1] if they are in [0, 255]
    original_frames = [frame.astype(np.float32) / 255.0 for frame in original_frames]
    predicted_frames = [frame.astype(np.float32) / 255.0 for frame in predicted_frames]

    for idx, (orig, pred) in enumerate(zip(original_frames, predicted_frames)):
        # Print debug information about the current frames
        print(f"Processing frame {idx + 1}/{len(original_frames)}")
        print(f"Original frame range: {orig.min()} to {orig.max()}")
        print(f"Predicted frame range: {pred.min()} to {pred.max()}")
        
        # Ensure the frames have the same shape
        assert orig.shape == pred.shape, f"Mismatch in frame shapes: {orig.shape} vs {pred.shape}"
        
        # Compute MSE for the current pair of frames
        mse = np.mean((orig - pred) ** 2)
        mse_values.append(mse)

    # Calculate the average MSE
    avg_mse = np.mean(mse_values)
    print(f"Average MSE across frames: {avg_mse}")
    return avg_mse


# Function to preprocess video frames
def preprocess_frames(frames, img_height, img_width):
    preprocessed_frames = [
        cv2.resize(frame, (img_width, img_height)) / 255.0 for frame in frames
    ]
    return np.array(preprocessed_frames)

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    cap.release()
    return frames

# Function to create a GIF
# Function to create a GIF
def create_gif(frames, gif_path, fps=10):
    print(f"Number of frames: {len(frames)}")
    print(f"Frame shape: {frames[0].shape if frames else 'No frames'}")
    try:
        # Ensure the directory for the GIF path exists
        gif_dir = os.path.dirname(gif_path)
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)

        # Process frames for GIF creation
        processed_frames = []
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[-1] == 1:  # If frame has a channel dimension
                frame = frame.squeeze()  # Remove the channel dimension
            frame = (frame * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
            processed_frames.append(frame)

        print(f"Creating GIF at {gif_path}...")
        imageio.mimsave(gif_path, processed_frames, fps=fps)
        print(f"GIF created successfully at {gif_path}")
    except Exception as e:
        print(f"Error while creating GIF: {e}")



def preprocess_video(video_path, seq_length, img_height, img_width, is_training=True):
    """
    Preprocess a video file by extracting and processing frames.

    Args:
        video_path (str): Path to the video file.
        seq_length (int): Required sequence length of frames.
        img_height (int): Height of the resized frames.
        img_width (int): Width of the resized frames.
        is_training (bool): Whether the function is used for training or inference.
    
    Returns:
        np.ndarray: Preprocessed video frames with shape (seq_length, img_height, img_width, 1).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < seq_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_width, img_height))  # Resize frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    
    cap.release()

    if len(frames) < seq_length:
        if is_training:
            # Discard videos with insufficient frames during training
            return None
        else:
            # Pad with empty frames for inference
            while len(frames) < seq_length:
                frames.append(np.zeros((img_height, img_width), dtype=np.uint8))

    # Normalize and reshape for model input
    frames = np.array(frames, dtype=np.float32) / 255.0  # Normalize pixel values
    frames = frames.reshape((seq_length, img_height, img_width, 1))  # Add channel dimension
    return frames

def process_video(video_path, img_height, img_width, input_length):
    """
    Process a video for inference and generate predictions.

    Args:
        video_path (str): Path to the video file.
        img_height (int): Height of the resized frames.
        img_width (int): Width of the resized frames.
        input_length (int): Maximum input sequence length for the model.

    Returns:
        tuple: Original frames, predicted frames, and runtime.
    """
    # Extract and preprocess frames
    preprocessed_frames = preprocess_video(video_path, SEQ_LENGTH, img_height, img_width, is_training=False)

    # Prepare input for the model
    input_data = np.expand_dims(preprocessed_frames, axis=0)  # Shape: (1, seq_length, img_height, img_width, 1)

    # Predict using the model
    start_time = time.time()
    predictions = model.predict(input_data)
    runtime = time.time() - start_time

    # Post-process predictions
    prediction_frames = [(frame.squeeze() * 255).astype(np.uint8) for frame in predictions[0]]
    print("Video processed successfully!")
    return preprocessed_frames, prediction_frames, runtime


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle video upload
        if 'video' not in request.files:
            return "No file uploaded", 400
        file = request.files['video']
        if file.filename == '':
            return "No file selected", 400

        # Save the uploaded video
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        # Model parameters
        IMG_HEIGHT, IMG_WIDTH = 64, 64  # Change if different
        INPUT_LENGTH = 10  # Adjust based on your model training

        # Process the video
        original_frames, prediction_frames, runtime = process_video(video_path, IMG_HEIGHT, IMG_WIDTH, INPUT_LENGTH)
        original_frames = [frame.squeeze() if frame.shape[-1] == 1 else frame for frame in original_frames]
        prediction_frames = [frame.squeeze() if frame.shape[-1] == 1 else frame for frame in prediction_frames]


        # Save GIFs for original and predictions
        results_dir = os.path.join(RESULTS_FOLDER, os.path.splitext(file.filename)[0])
        os.makedirs(results_dir, exist_ok=True)
        current_dir = os.getcwd()

        # Set paths for the GIFs in the current directory
        original_gif_path = os.path.join('static', 'original.gif')
        prediction_gif_path = os.path.join('static', 'prediction.gif')
        avg_ssim = calculate_ssim(original_frames, prediction_frames)
        avg_mse = calculate_mse(original_frames, prediction_frames)


        print(f"Original GIF Path: {original_gif_path}")
        print(f"Prediction GIF Path: {prediction_gif_path}")



        create_gif(original_frames, original_gif_path)
        create_gif(prediction_frames, prediction_gif_path)

        return render_template('results.html',
                               runtime=runtime,
                               ssim=avg_ssim,
                               mse=avg_mse)
    return render_template('index.html')

@app.route('/results/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)

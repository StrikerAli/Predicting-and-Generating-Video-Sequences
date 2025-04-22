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

class PredRNNCell(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride=1):
        super(PredRNNCell, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_x = layers.Conv2D(
            self.filters, self.kernel_size, strides=self.stride, padding="same", activation="relu"
        )
        self.conv_h = layers.Conv2D(
            self.filters, self.kernel_size, strides=self.stride, padding="same", activation="relu"
        )

    def build(self, input_shape):
        # Initialize Conv2D layers based on input shape
        self.conv_x.build(input_shape)
        self.conv_h.build((None, input_shape[1], input_shape[2], self.filters))  # Hidden state shape
        super(PredRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        x, h = inputs, states
        if h is None:
            # Initialize hidden state with correct shape
            batch_size, height, width, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.filters
            h = tf.zeros((batch_size, height, width, self.filters), dtype=tf.float32)
        xh = self.conv_x(x)
        hh = self.conv_h(h)
        h_next = tf.nn.relu(xh + hh)
        return h_next, h_next

class PredRNN(tf.keras.Model):
    def __init__(self, input_shape, hidden_size, output_channels, num_layers=3):
        super(PredRNN, self).__init__()
        self.layers_list = [PredRNNCell(hidden_size, (3, 3)) for _ in range(num_layers)]
        self.conv_output = layers.Conv2D(output_channels, (3, 3), padding="same")

    def build(self, input_shape):
        time_steps, height, width, channels = input_shape[1:]
        for layer in self.layers_list:
            layer.build((None, height, width, channels))
            channels = layer.filters  # Update channels for the next layer
        self.conv_output.build((None, height, width, channels))
        super(PredRNN, self).build(input_shape)

    def call(self, inputs):
        # Dynamically determine the batch size and initialize states
        batch_size = tf.shape(inputs)[0]
        time_steps, height, width = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        channels = inputs.shape[4]  # Initial input channels
        
        # Initialize states dynamically
        states = [None] * len(self.layers_list)

        outputs = []
        for t in range(time_steps):
            x = inputs[:, t]
            for i, layer in enumerate(self.layers_list):
                if states[i] is None:
                    states[i] = tf.zeros((batch_size, height, width, layer.filters), dtype=tf.float32)
                x, states[i] = layer(x, states[i])
            outputs.append(x)
        outputs = tf.stack(outputs, axis=1)  # Shape: (batch_size, time_steps, height, width, channels)

        # Update channels to match the filters of the last layer
        channels = self.layers_list[-1].filters

        # Reshape to combine batch_size and time_steps
        outputs = tf.reshape(outputs, (-1, height, width, channels))  # Shape: (batch_size * time_steps, height, width, channels)
        
        # Apply the final Conv2D layer
        outputs = self.conv_output(outputs)  # Shape: (batch_size * time_steps, height, width, output_channels)
        
        # Reshape back to include the time dimension
        outputs = tf.reshape(outputs, (batch_size, time_steps, height, width, -1))  # Shape: (batch_size, time_steps, height, width, output_channels)
        
        return outputs


# Define the model path
# Build the model
input_shape = (INPUT_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
output_channels = IMG_CHANNEL
hidden_size = 64

model = PredRNN(input_shape[1:], hidden_size, output_channels)
model.build((None,) + input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")



# Load the model with the updated custom objects

print("Model loaded built!")

model_weights_path = 'C:\\Users\\aalia\\Downloads\\rnn\\rnn\\predrnn_model_weights.weights.h5'
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
    mse_values = []
    for orig, pred in zip(original_frames, predicted_frames):
        mse = np.mean((orig - pred) ** 2)
        mse_values.append(mse)
    return np.mean(mse_values)


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

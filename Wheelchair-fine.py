import time
import asyncio
import numpy as np
import websockets
from sklearn.svm import OneClassSVM
from neuropy3.neuropy3 import MindWave
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
from collections import deque
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import os

app = Flask(__name__)
CORS(app)

# WebSocket URI for ESP32
ESP32_URI = "ws://172.20.10.2:81"

# Define EEG features
FEATURES = ['delta', 'theta', 'alpha_l', 'alpha_h', 'beta_l', 'beta_h']

# Global variables
last_function_call_time = time.time()
latest_attention_value = 0
signal_processor = None
last_state = None

class EEGModelManager:
    def __init__(self, window_size=1000, fine_tune_threshold=100, validation_size=0.2):
        self.model = None
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.fine_tune_threshold = fine_tune_threshold
        self.validation_size = validation_size
        self.training_buffer = deque(maxlen=window_size)
        self.performance_history = []
        self.checkpoint_dir = "model_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def preprocess_features(self, features):
        if not self.training_buffer:
            self.scaler.fit(features.reshape(1, -1))
        return self.scaler.transform(features.reshape(1, -1))
    
    def initial_training(self, initial_data):
        if len(initial_data) >= 10:
            X = np.array(initial_data)
            X = self.scaler.fit_transform(X)
            X_train, X_val = train_test_split(X, test_size=self.validation_size)
            
            self.model = OneClassSVM(nu=0.1, kernel='rbf')
            self.model.fit(X_train)
            
            val_score = self.model.score_samples(X_val).mean()
            self.performance_history.append(val_score)
            
            self.save_checkpoint("initial")
            print("Initial model trained. Validation score:", val_score)
            return True
        return False
    
    def fine_tune(self):
        if len(self.training_buffer) >= self.fine_tune_threshold:
            print("Starting fine-tuning...")
            
            X = np.array(list(self.training_buffer))
            X = self.scaler.transform(X)
            X_train, X_val = train_test_split(X, test_size=self.validation_size)
            
            new_model = OneClassSVM(nu=0.1, kernel='rbf')
            new_model.fit(X_train)
            
            new_val_score = new_model.score_samples(X_val).mean()
            
            if not self.performance_history or new_val_score > self.performance_history[-1]:
                self.model = new_model
                self.performance_history.append(new_val_score)
                self.save_checkpoint("fine_tuned")
                print(f"Model fine-tuned successfully. New validation score: {new_val_score}")
                return True
            else:
                print("Fine-tuning skipped: No improvement in performance")
                return False
    
    def predict(self, features):
        if self.model is None:
            return None
        
        features_processed = self.preprocess_features(features)
        prediction = self.model.predict(features_processed)
        
        self.training_buffer.append(features.flatten())
        
        if len(self.training_buffer) >= self.fine_tune_threshold:
            self.fine_tune()
        
        return prediction[0]
    
    def save_checkpoint(self, checkpoint_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{checkpoint_type}_model_{timestamp}.joblib"
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'performance_history': self.performance_history
        }
        
        joblib.dump(checkpoint, path)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        self.model = checkpoint['model']
        self.scaler = checkpoint['scaler']
        self.performance_history = checkpoint['performance_history']
        print(f"Model loaded from checkpoint: {checkpoint_path}")

class AttentionSignalProcessor:
    def __init__(self, 
                 window_size=10,
                 median_window=5,
                 threshold_change=20,
                 min_stable_duration=1.0):
        self.window_size = window_size
        self.median_window = median_window
        self.threshold_change = threshold_change
        self.min_stable_duration = min_stable_duration
        
        self.attention_buffer = deque(maxlen=window_size)
        self.last_filtered_value = None
        self.last_state_change = 0
        self.current_state = None
    
    def moving_average_filter(self, value):
        self.attention_buffer.append(value)
        return np.mean(self.attention_buffer)
    
    def median_filter(self, values):
        return float(medfilt(np.array(values), self.median_window)[0])
    
    def hysteresis_filter(self, value):
        current_time = time.time()
        new_state = 'MOVE' if value > 80 else 'STAY'
        
        if (self.current_state != new_state and 
            current_time - self.last_state_change >= self.min_stable_duration):
            self.current_state = new_state
            self.last_state_change = current_time
            return new_state
        
        return self.current_state
    
    def process_attention(self, value):
        ma_value = self.moving_average_filter(value)
        
        if len(self.attention_buffer) >= self.median_window:
            med_value = self.median_filter(list(self.attention_buffer))
        else:
            med_value = ma_value
        
        if self.last_filtered_value is not None:
            change = med_value - self.last_filtered_value
            if abs(change) > self.threshold_change:
                med_value = self.last_filtered_value + np.sign(change) * self.threshold_change
        
        self.last_filtered_value = med_value
        state = self.hysteresis_filter(med_value)
        
        return med_value, state

async def send_command_via_websocket(command):
    try:
        async with websockets.connect(ESP32_URI) as websocket:
            await websocket.send(command)
            print(f"Sent command via WebSocket: {command}")
    except Exception as e:
        print(f"Failed to send command via WebSocket: {e}")

def MOVE():
    print("Function MOVE: Attention is high (more than 80)")
    asyncio.run(send_command_via_websocket('MOVE'))

def STAY():
    print("Function STAY: Attention is low (less than or equal to 80)")
    asyncio.run(send_command_via_websocket('STAY'))

def extract_features(eeg_data):
    return np.array([eeg_data[f] for f in FEATURES]).reshape(1, -1)

def eeg_callback(data, model_manager):
    features = extract_features(data)
    
    if model_manager.model is None:
        initial_data = []
        initial_data.append(features.flatten())
        
        if model_manager.initial_training(initial_data):
            print("Initial model training completed")
    else:
        prediction = model_manager.predict(features)
        
        if prediction == 1:
            print(f"EEG: {data}")
        else:
            print("EEG data classified as erratic, ignoring.")

def meditation_callback(value):
    print("Meditation: ", value)

def attention_callback(value):
    global last_function_call_time, latest_attention_value, signal_processor, last_state
    
    filtered_value, state = signal_processor.process_attention(value)
    latest_attention_value = filtered_value
    
    print(f"Raw Attention: {value}, Filtered: {filtered_value:.1f}, State: {state}")
    
    current_time = time.time()
    if current_time - last_function_call_time >= 1 and state != last_state:
        if state == 'MOVE':
            MOVE()
        else:
            STAY()
        last_function_call_time = current_time
        last_state = state

@app.route('/get_attention', methods=['GET'])
def get_attention():
    return jsonify({"attention": latest_attention_value}), 200

@app.route('/get_model_performance', methods=['GET'])
def get_model_performance():
    if hasattr(model_manager, 'performance_history'):
        return jsonify({
            "performance_history": model_manager.performance_history,
            "latest_score": model_manager.performance_history[-1] if model_manager.performance_history else None
        }), 200
    return jsonify({"error": "Model performance data not available"}), 404

def run_flask():
    app.run(debug=False, port=5000)

def main():
    global last_function_call_time, signal_processor, model_manager

    model_manager = EEGModelManager(
        window_size=1000,
        fine_tune_threshold=100,
        validation_size=0.2
    )

    signal_processor = AttentionSignalProcessor(
        window_size=10,
        median_window=5,
        threshold_change=20,
        min_stable_duration=1.0
    )

    #Counter the first STAY command
    print("Sending initial MOVE command...")
    MOVE()
    last_state = 'MOVE'
    
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    mw = MindWave(address='A4:DA:32:70:03:4E', autostart=False, verbose=3)

    mw.set_callback('eeg', lambda data: eeg_callback(data, model_manager))
    mw.set_callback('meditation', meditation_callback)
    mw.set_callback('attention', attention_callback)

    mw.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    finally:
        print("Cleaning up...")
        mw.stop()
        if model_manager.model is not None:
            model_manager.save_checkpoint("final")

if __name__ == "__main__":
    main()
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os
from tensorflow import keras

app = Flask(__name__)

# Global model variable
model = None

def load_saved_model():
    """Load the pre-trained model"""
    global model
    
    try:
        print("=" * 50)
        print("LOADING TRAINED MODEL FOR RAILWAY")
        print("=" * 50)
        
        model_path = 'digit_model.h5'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files available: {os.listdir('.')}")
            return None
        
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model loaded successfully!")
        
        # Test the model
        test_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = model.predict(test_input, verbose=0)
        print("✓ Model test successful!")
        
        print("=" * 50)
        print("MODEL READY FOR PREDICTIONS!")
        print("=" * 50)
        
        return model
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Load model when module imports
print("🚀 Initializing Railway application...")
load_saved_model()

def preprocess_image(image_data):
    """Preprocess uploaded image for prediction"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply thresholding
        img = cv2.adaptiveThreshold(
            img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Find contours and crop
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            digit = img[y:y+h, x:x+w]
        else:
            digit = img
        
        # Resize maintaining aspect ratio
        h, w = digit.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_w, new_h = 20, max(1, int(20 * h / w))
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28
        final_img = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize and reshape
        final_img = final_img.astype('float32') / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please refresh the page.'
            })
        
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data'})
        
        # Preprocess and predict
        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit]) * 100
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        print(f"✓ Prediction: {predicted_digit}, Confidence: {confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'platform': 'Railway'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

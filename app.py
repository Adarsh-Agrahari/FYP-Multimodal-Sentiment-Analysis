from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import pickle
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO in production
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='error.log',  # Log errors to this file
    filemode='a'  # Append mode
)
logger = logging.getLogger()

# Load the pickled model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'multimodal_sentiment_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
    return model, device

# Initialize tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize model
try:
    model, device = load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.critical(f"Error loading model: {str(e)}", exc_info=True)
    model, device = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logger.error("Model is not loaded. Request cannot be processed.")
        return jsonify({
            'success': False,
            'error': 'Model not loaded properly. Please check the server logs for details.'
        }), 500

    try:
        # Validate input
        if 'text' not in request.form or 'image' not in request.files:
            logger.warning("Missing text or image input in the request.")
            return jsonify({
                'success': False,
                'error': 'Missing text or image input.'
            }), 400

        text = request.form['text']
        image_file = request.files['image']

        # Process image
        try:
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0).to(device)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f"Error processing image: {str(e)}"
            }), 400

        # Process text
        try:
            encoded_text = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoded_text['input_ids'].to(device)
            attention_mask = encoded_text['attention_mask'].to(device)
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f"Error processing text: {str(e)}"
            }), 400

        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f"Error during prediction: {str(e)}"
            }), 500

        # Map prediction to sentiment
        sentiments = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiments.get(predicted_class, "Unknown")
        probs = probabilities[0].cpu().numpy().tolist()

        logger.info(f"Prediction successful. Sentiment: {sentiment}")
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'probabilities': {
                'negative': round(probs[0], 4),
                'neutral': round(probs[1], 4),
                'positive': round(probs[2], 4)
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

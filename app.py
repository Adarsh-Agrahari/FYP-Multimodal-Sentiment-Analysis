# app.py
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import pickle
import io
import os

app = Flask(__name__)

# Load the pickled model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('multimodal_sentiment_model.pkl', 'rb') as f:
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
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model, device = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded properly'
        })
    
    try:
        # Get text and image from request
        text = request.form['text']
        image_file = request.files['image']
        
        # Process image
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        
        # Process text
        encoded_text = tokenizer(text, padding='max_length', truncation=True, 
                               max_length=128, return_tensors='pt')
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Map prediction to sentiment
        sentiments = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiments[predicted_class]
        probs = probabilities[0].cpu().numpy().tolist()
        
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'probabilities': {
                'negative': probs[0],
                'neutral': probs[1],
                'positive': probs[2]
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
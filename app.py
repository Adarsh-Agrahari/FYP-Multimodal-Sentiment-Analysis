import logging
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import torchvision
import transformers
import pickle
import os
import torch.nn as nn
import traceback
# Initialize logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MultimodalSentimentModel(nn.Module):
    def __init__(self, bert_model, resnet_model, num_classes):
        super(MultimodalSentimentModel, self).__init__()
        self.text_model = bert_model
        self.image_model = resnet_model
        self.text_output_size = 768
        self.image_output_size = 2048
        self.fc1 = nn.Linear(self.text_output_size + self.image_output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, image):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]
        image_output = self.image_model(image)
        combined = torch.cat((text_output, image_output), dim=1)
        x = self.fc1(combined)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

app = Flask(__name__)
def load_model():
    device = torch.device("cpu")
    try:
        # Attempt to load the checkpoint
        checkpoint = torch.load('model.pkl', map_location=device)
        
        # Log checkpoint keys for debugging
        logging.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Ensure required keys are present
        required_keys = ['model_state_dict', 'bert_model', 'num_classes']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' is missing in the checkpoint.")
        
        # Initialize models
        bert_model = transformers.AutoModel.from_pretrained(checkpoint['bert_model'])
        resnet_model = torchvision.models.resnet50(pretrained=True)
        resnet_model.fc = nn.Identity()  # Modify ResNet to output raw features
        
        # Create multimodal model
        model = MultimodalSentimentModel(bert_model, resnet_model, checkpoint['num_classes'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logging.info("Model loaded successfully.")
        return model, device

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error("Traceback:\n" + traceback.format_exc())
        return None, None

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
    logging.error(f"Error initializing model: {str(e)}")
    model, device = None, None

@app.route('/')
def home():
    logging.info("Home route accessed.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        error_message = "Model not loaded properly."
        logging.error(f"{error_message} Exception details: {error_message}")
        # Log the full exception traceback for more details
        logging.error("Error details:\n" + traceback.format_exc())
        return jsonify({
            'success': False,
            'error': error_message
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
        
        logging.info(f"Prediction made: {sentiment}, probabilities: {probs}")
        
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
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')

class MedicalDataset:
    """Base class for handling different medical datasets"""
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self):
        raise NotImplementedError
        
    def preprocess(self):
        raise NotImplementedError

class BasicMedicalDataset(MedicalDataset):
    """Handler for AI_SET_1: Basic medical symptoms dataset"""
    def load_data(self):
        self.data = pd.read_csv('AI_SET_1.csv')
        return self
        
    def preprocess(self):
        # Define binary columns
        binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
        
        # Convert 'Yes'/'No' values to 1/0 for binary columns
        for col in binary_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].map({'Yes': 1, 'No': 0})
        
        # Encode 'Disease' column if it exists
        if 'Disease' in self.data.columns:
            le = LabelEncoder()
            self.data['Disease'] = le.fit_transform(self.data['Disease'])
        
        # Convert 'Outcome_Variable' to binary (Positive = 1, others = 0)
        if 'Outcome_Variable' in self.data.columns:
            self.data['Outcome_Variable'] = self.data['Outcome_Variable'].apply(lambda x: 1 if x == 'Positive' else 0)
        
        # Optional: Handle missing values (example: fill with mode or drop rows)
        self.data.fillna(self.data.mode().iloc[0], inplace=True)
        
        return self

class ComprehensiveMedicalDataset(MedicalDataset):
    """Handler for AI_SET_3: Comprehensive symptoms dataset"""
    def load_data(self):
        
        self.data = pd.read_csv('AI_SET_3.csv')
        return self
        
    def preprocess(self):
        # Label encode the target variable 'prognosis'
        le = LabelEncoder()
        self.data['prognosis'] = le.fit_transform(self.data['prognosis'])
        
        # Fill missing values (if any) - could use median, mean, or mode, or drop rows
        self.data.fillna(self.data.mode().iloc[0], inplace=True)
        
        # Check if any categorical features (other than 'prognosis') need encoding (your data seems to be binary, so not needed)
        
        # Scaling/Normalizing the features - for binary columns, this may not be necessary
        # But you might want to scale the features if using algorithms like SVM or k-NN
        features = self.data.drop('prognosis', axis=1)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Update the data with scaled features
        self.data_scaled = pd.DataFrame(scaled_features, columns=features.columns)
        self.data_scaled['prognosis'] = self.data['prognosis']
        
        return self.data_scaled

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Medibot:
    def __init__(self):
        self.basic_dataset = BasicMedicalDataset()
        self.comprehensive_dataset = ComprehensiveMedicalDataset()
        self.ffn_model = None
        self.scaler = StandardScaler()
        self.confidence_threshold = 0.3  # Lowered threshold for more predictions
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer()
        self.symptom_embeddings = None
    
    def initialize(self):
        print("Initializing Medibot...")
        self.comprehensive_dataset.load_data().preprocess()
        self._train_ffn()
        self._prepare_symptom_embeddings()
        print("Initialization complete!")
    
    def _prepare_symptom_embeddings(self):
        symptoms = self.comprehensive_dataset.data.columns[:-1]  # Exclude 'prognosis'
        self.symptom_embeddings = self.vectorizer.fit_transform(symptoms)
        
    def _train_ffn(self):
        # Prepare data for FFN
        X = self.comprehensive_dataset.data.drop('prognosis', axis=1).values
        y = self.comprehensive_dataset.data['prognosis'].values
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # Initialize the model
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = len(np.unique(y))
        self.ffn_model = FeedForwardNet(input_size, hidden_size, output_size)
        
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.ffn_model.parameters(), lr=0.001)
        
        epochs = 100
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.ffn_model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate the model
        with torch.no_grad():
            test_outputs = self.ffn_model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Test Accuracy: {accuracy:.4f}')
    
    def predict_disease(self, symptoms):
        # Initialize input data array with zeros
        input_data = np.zeros(len(self.comprehensive_dataset.data.columns) - 1)
        
        # Populate the input_data array with 1 for each symptom found in the dataset
        for symptom in symptoms:
            if symptom in self.comprehensive_dataset.data.columns:
                input_data[self.comprehensive_dataset.data.columns.get_loc(symptom)] = 1
        
        # Standardize the input data using the scaler
        input_data = self.scaler.transform([input_data])
        input_tensor = torch.FloatTensor(input_data)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            output = self.ffn_model(input_tensor)
            # Use softmax to get class probabilities and get the class with the highest probability
            confidence, predicted = torch.max(F.softmax(output, dim=1), 1)
        
        # Convert the predicted index to a disease name
        disease_idx = predicted.item()
        confidence = confidence.item()
        
        # Map the disease index to the disease name from the 'prognosis' column
        disease_name = self.comprehensive_dataset.data['prognosis'].unique()[disease_idx]
        
        return disease_name, confidence
    
    
    def extract_symptoms(self, user_input: str) -> list:
        doc = self.nlp(user_input)
        extracted_symptoms = []
        
        # Extract noun phrases and named entities
        potential_symptoms = [chunk.text.lower() for chunk in doc.noun_chunks] + [ent.text.lower() for ent in doc.ents]
        
        # Match extracted phrases with known symptoms using TF-IDF similarity
        user_vector = self.vectorizer.transform(potential_symptoms)
        similarities = (user_vector * self.symptom_embeddings.T).toarray()
        
        for i, phrase in enumerate(potential_symptoms):
            max_sim = similarities[i].max()
            if max_sim > 0.5:  # Adjust this threshold as needed
                matched_symptom = self.comprehensive_dataset.data.columns[similarities[i].argmax()]
                extracted_symptoms.append(matched_symptom)
        
        return list(set(extracted_symptoms))  # Remove duplicates
    
    def process_input(self, user_input: str) -> str:
        symptoms = self.extract_symptoms(user_input)
        
        if not symptoms:
            return "I couldn't identify any specific symptoms. Could you please describe how you're feeling in more detail?"
        
        predicted_disease, confidence = self.predict_disease(symptoms)
        
        response = f"Based on the symptoms you've described ({', '.join(symptoms)}), "
        
        if confidence >= self.confidence_threshold:
            response += f"it's possible you might have {predicted_disease}. "
            response += f"Confidence: {confidence * 100:.2f}%. "
        else:
            response += "I'm not confident enough to suggest a specific condition. "
        
        response += "Please consult with a healthcare professional for an accurate diagnosis."
        
        if len(symptoms) < 3:
            response += " Are there any other symptoms you're experiencing?"
        
        return response

def main():
    medibot = Medibot()
    medibot.initialize()
    
    print("Medibot: Hello! I'm your virtual healthcare assistant. How can I help you today?")

    test_inputs = [
    "I have a headache and fever",
    "My stomach hurts and I feel nauseous",
    "I'm experiencing shortness of breath",
    "I have a sore throat and runny nose",
    "I'm feeling very tired lately"
]

    for input_text in test_inputs:
        print(f"\nUser: {input_text}")
        response = medibot.process_input(input_text)
        print(f"Medibot: {response}")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Medibot: Take care! Goodbye!")
            break
            
        response = medibot.process_input(user_input)
        print("Medibot:", response)

if __name__ == "__main__":
    main()

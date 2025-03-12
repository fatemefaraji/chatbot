import tensorflow as tf
from tensorflow import keras
import numpy as np
import sentencepiece as spm
import json
import os
from typing import Dict, List, Optional, Tuple

class DatabaseManager:
    def __init__(self, data_path: str):
        with open(os.path.join(data_path, "processed_slots.json"), "r", encoding="utf-8") as f:
            self.slots = json.load(f)
        with open(os.path.join(data_path, "processed_ontology.json"), "r", encoding="utf-8") as f:
            self.ontology = json.load(f)
        with open(os.path.join(data_path, "processed_database.json"), "r", encoding="utf-8") as f:
            self.database = json.load(f)

    def extract_slots(self, text: str) -> Dict[str, str]:
        slots_found = {}
        text = text.lower()

        categories = ['hotel', 'restaurant', 'train', 'attraction', 'taxi', 'bus', 'hospital']
        category = next((cat for cat in categories if cat in text), None)
        
        if category:
            for slot in self.slots.get(category, []):
                slot_values = self.ontology.get(category, {}).get(f"{category}-semi-{slot.split('-')[-1]}", [])
                for value in slot_values:
                    if value.lower() in text:
                        slots_found[slot] = value
        
        return slots_found

    def query_database(self, category: str, slots: Dict[str, str]) -> List[Dict]:
        results = []
        entries = self.database.get(category, [])
        
        for entry in entries:
            matches = True
            for slot, value in slots.items():
                if slot.split('-')[-1] in entry:
                    if entry[slot.split('-')[-1]].lower() != value.lower():
                        matches = False
                        break
            if matches:
                results.append(entry)
        
        return results

class EnhancedChatbot:
    def __init__(self, model_path: str, tokenizer_path: str, data_path: str):
        self.max_sequence_length = 128
        
        self.model = keras.models.load_model(model_path, custom_objects={
            "Reformer": Reformer, 
            "ReformerBlock": ReformerBlock, 
            "ReversibleResidualLayer": ReversibleResidualLayer, 
            "LSHSelfAttention": LSHSelfAttention, 
            "FeedForward": FeedForward, 
            "TokenEmbedding": TokenEmbedding
        })
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.db_manager = DatabaseManager(data_path)

    def preprocess_input(self, text: str) -> np.ndarray:
        """Preprocess input text for the model."""
        tokens = self.tokenizer.encode_as_ids(text)
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
        padded_tokens = np.zeros((1, self.max_sequence_length), dtype=np.int32)
        padded_tokens[0, :len(tokens)] = tokens
        return padded_tokens

    def postprocess_output(self, predictions: np.ndarray) -> str:
        predicted_ids = np.argmax(predictions[0], axis=-1)
        if self.tokenizer.eos_id() in predicted_ids:
            predicted_ids = predicted_ids[:np.where(predicted_ids == self.tokenizer.eos_id())[0][0]]
        return self.tokenizer.decode_ids(predicted_ids.tolist())

    def format_database_response(self, category: str, results: List[Dict]) -> str:
        if not results:
            return f"I couldn't find any {category} matching your criteria."
        
        response = f"I found {len(results)} matching {category}(s):\n"
        for i, result in enumerate(results, 1):
            response += f"\n{i}. "
            if 'name' in result:
                response += f"{result['name']}"
            if 'type' in result:
                response += f" ({result['type']})"
            if 'area' in result:
                response += f" in {result['area']}"
            if 'pricerange' in result:
                response += f" - {result['pricerange']} price range"
            if 'phone' in result:
                response += f"\n   phone: {result['phone']}"
            if 'address' in result:
                response += f"\n   address: {result['address']}"
        
        return response

    def handle_query(self, user_input: str) -> str:
        slots = self.db_manager.extract_slots(user_input)
        
        if slots:
            category = next((cat for cat in self.db_manager.slots.keys() 
                           if any(slot.startswith(cat) for slot in slots)), None)
            if category:
                results = self.db_manager.query_database(category, slots)
                if results:
                    return self.format_database_response(category, results)

        try:
            preprocessed_input = self.preprocess_input(user_input)
            print(f"preprocessed input: {preprocessed_input}")  
            with tf.device('/CPU:0'):
                prediction = self.model.predict(preprocessed_input, verbose=0)
            print(f"model prediction: {prediction}")  
            return self.postprocess_output(prediction)
        except Exception as e:
            return f"I apologize, but Im facing an error: {str(e)}"

    def chat(self):
        print("welcome! Type 'exit' to end the chat.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye! have a nice day !")
                    break
                    
                if not user_input:
                    print("please type something!")
                    continue
                
                response = self.handle_query(user_input)
                print(f"assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nChat ended by user.")
                break
            except Exception as e:
                print(f"error occurred: {str(e)}")
                print("try again!")

if __name__ == "__main__":
    MODEL_PATH = 'reformer_final_model.keras'
    TOKENIZER_PATH = "/content/drive/MyDrive/Week 4/data/tokenizer.model"
    DATA_PATH = "/content/drive/MyDrive/Week 4/data"
    
    chatbot = EnhancedChatbot(MODEL_PATH, TOKENIZER_PATH, DATA_PATH)
    chatbot.chat()
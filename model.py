import pickle as pk
import numpy as np
import spacy
import torch
import random
from sklearn.tree import _tree
import Searching_des as des
from reply import all_response_msg
from nn_model import NeuralNet

#******************************************* for Device ***********************************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Remove warnings
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# =====================================================  LOAD MODELS ==================================================
dtc , le  = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/chatbot_modelsave','rb'))
model,second_le  = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/chatbot_model','rb'))
ners = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/newsave_model','rb'))

# Load NN Model 
FILE = "/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/data.pth"
data = torch.load(FILE)

# Initialize Global Variables
return_input_disease = []
disease_first = []
symptoms_get = []
feature_names = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes']

nlp = spacy.load("en_core_web_sm")

#============================================== STEP 2 ===================================================
# Predicted Disease Step 2
def input_output(present_disease):
    value_dicts = {
        'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
        'Diabetes': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level']
    }
    symptoms_given = value_dicts.get(present_disease[0], [])
    symptoms_get.clear()
    symptoms_get.extend(symptoms_given)

#============================================== Bag of Words ===================================================
def bag_of_words(tokenize_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0
    return bag

# Load Neural Network Model 
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

nn_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nn_model.load_state_dict(model_state)
nn_model.eval()

# NLTK Model Output
def nltk_output(msg):
    sentence = [token.lemma_ for token in nlp(msg)]
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = nn_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        return random.choice(all_response_msg[tag])
    return "I'm not sure. Could you please provide more details?"

# ================================================== TREE RECURSION ==============================================
# Step 1-C
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    disease_first.clear() 
    disease_first.append(disease)
    input_output(disease)

# Step 1-B 
def recurse(node, depth):
    global return_input_disease
    tree = dtc.tree_
    feature = tree.feature[node]
    threshold = tree.threshold[node]
    
    if threshold != -2:
        print(f"Do you have {feature_names[feature]}? (yes/no)")
        ans = input().strip().lower()
        if ans == 'yes':
            recurse(tree.children_left[node], depth + 1)
        else:
            recurse(tree.children_right[node], depth + 1)
    else:
        return_input_disease.append(tree.value[node])
        print_disease(return_input_disease)
        
# ================================================== Chatbot Loop ==============================================
def chatbot():
    print("Hello! I am your HealthCare ChatBot. How can I assist you today?")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit", "bye"]:
            print("ChatBot: Take care! Goodbye.")
            break
        
        response = nltk_output(msg)
        print(f"ChatBot: {response}")
        
        if "symptom" in msg:
            print("ChatBot: Let's check for potential diseases.")
            recurse(0, 1)

# Start Chatbot
if __name__ == "__main__":
    chatbot()

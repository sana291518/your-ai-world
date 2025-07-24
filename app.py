from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from difflib import SequenceMatcher
from flask import render_template_string
import json
import time
import cv2
import numpy as np
from keras.models import model_from_json
import threading
import classifier

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

face_detector = cv2.CascadeClassifier("emotion_detector/haarcascade_frontalface_default.xml")

# Load the Model and Weights
emotion_model = model_from_json(open("emotion_detector/facial_expression_model_structure.json", "r").read())
emotion_model.load_weights('emotion_detector/facial_expression_model_weights.h5')
emotion_model.make_predict_function()

# socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for demonstration


# @socketio.on('connect', namespace='/realtime_emotion_detection')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect', namespace='/realtime_emotion_detection')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('emotion_data', namespace='/realtime_emotion_detection')
# def handle_emotion_data(data):
#     # Perform emotion detection on the received video frames (data['imageData'])
#     # Send the detected emotion data back to the client
#     emit('emotion_result', {'emotion': 'happy'})  # Replace 'happy' with the actual emotion result

# Load your facial emotion detection model and setup the webcam
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("facialemotionmodel.h5")

# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)


# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0

# # Initialize webcam as a global variable
# webcam = None
# labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function for real-time emotion detection
# def real_time_emotion_detection():
#     global webcam
#     while True:
#         if webcam is not None:
#             ret, im = webcam.read()
#             if not ret:
#                 break

#             gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(im, 1.3, 5)

#             try:
#                 for (p, q, r, s) in faces:
#                     image = gray[q:q + s, p:p + r]
#                     cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
#                     image = cv2.resize(image, (48, 48))
#                     img = extract_features(image)
#                     pred = model.predict(img)
#                     prediction_label = labels[pred.argmax()]
#                     cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
#                                 (0, 0, 255))
#                 cv2.imshow("Output", im)

#                 # Check for key press (27 is the ASCII value for ESC)
#                 key = cv2.waitKey(1)
#                 if key == 27:
#                     break

#             except cv2.error as e:
#                 print("Error:", e)

#     # Release webcam when the thread is terminated
#     webcam.release()
#     cv2.destroyAllWindows()

# Create a thread for real-time emotion detection
# emotion_detection_thread = threading.Thread(target=real_time_emotion_detection)

# Add a global variable to track camera status
# camera_started = False

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load dataset from JSON file
with open('data/dataset.json', 'r') as json_file:
    dataset = json.load(json_file)

# Define the new dataset format
class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        user_input = self.conversations[idx]['user']
        ai_response = self.conversations[idx]['assistant']

        user_input_tokens = self.tokenizer.encode(user_input, truncation=True, max_length=self.max_length)
        ai_response_tokens = self.tokenizer.encode(ai_response, truncation=True, max_length=self.max_length)

        return user_input_tokens, ai_response_tokens

# Define the collate_fn function
def collate_fn(batch):
    input_ids, labels = zip(*batch)

    # Find the maximum length in the batch
    max_len = max(max(len(ids) for ids in input_ids), max(len(lbls) for lbls in labels))

    # Pad input_ids and labels to have the same length in each batch
    input_ids_padded = torch.stack([torch.cat([torch.tensor(ids, dtype=torch.long), torch.zeros(max_len - len(ids), dtype=torch.long)], dim=0) for ids in input_ids])
    labels_padded = torch.stack([torch.cat([torch.tensor(lbls, dtype=torch.long), torch.zeros(max_len - len(lbls), dtype=torch.long)], dim=0) for lbls in labels])

    return input_ids_padded, labels_padded

# After initializing your dataset
conversation_dataset = ConversationDataset(dataset, tokenizer)

# Check if the dataset is empty
if len(conversation_dataset) == 0:
    print("Dataset is empty.")
else:
    # Continue with DataLoader initialization
    dataloader = DataLoader(conversation_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Placeholder for cute messages and jokes
    cute_messages = [
        "i ap-peach-iate you 🧸",
        "don't be sadish have a radish 🤤",
        "I carrot alot about you 🥕",
        "it's okay to put yourself first 😘",
        "BREAKING NEWS! You're super cute 💞"
    ]

    jokes = [
        "what do you call a bear with no ears? B(heheheh)",
        "What do you call an ant that wont go away? 🐜 permanant",
        "Why did the maths book look so sad?😔 it was full of problems🤭",
        "What do you call a man with no body and just a nose👃 nobody nose🕺",
        "How do you make seven an even number?🤔 take the s out(heheheh)"
    ]

    # Set up Flask routes
    @app.route('/')
    def home():
        return render_template('layout.html')

    @app.route('/chat')
    def chat():
        return render_template('chat.html')

# Update your realtime_emotion_detection route
@app.route('/realtime_emotion_detection')
def realtime_emotion_detection():
    return render_template('realtime_emotion_detection.html')

@app.route('/uploade', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # f.save("somefile.jpeg")
        # f = request.files['file']

        f = request.files['file'].read()
        npimg = np.fromstring(f, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        face_properties = classifier.classify(img, face_detector, emotion_model)

        return json.dumps(face_properties)

# Update your realtime_emotion_detection route
# @app.route('/realtime_emotion_detection')
# def realtime_emotion_detection():
#     global camera_started, webcam
#     if request.args.get('start_camera') == 'true':
#         # Initialize the webcam if it's not already started
#         if webcam is None:
#             webcam = cv2.VideoCapture(0)
#         camera_started = True
#         return render_template_string('realtime_emotion_detection.html', ws_url=request.url.replace('http', 'ws'))

#     # Release webcam if the user leaves the page
#     elif camera_started and request.args.get('start_camera') != 'true':
#         webcam.release()
#         camera_started = False

    # return render_template('realtime_emotion_detection.html')
@app.route('/get_cute_message', methods=['GET'])
def get_cute_message():
        cute_message = cute_messages.pop(0) if cute_messages else "No more cute messages!"
        return jsonify({'cute_message': cute_message})

@app.route('/get_joke', methods=['GET'])
def get_joke():
        joke = jokes.pop(0) if jokes else "No more jokes!"
        return jsonify({'joke': joke})

    # Rest of the code remains unchanged...

@app.route('/process_chat', methods=['POST'])
def process_chat():
        try:
            user_input = request.json.get('user_input')

            print(f"Received user input: {user_input}")

            if any(emotion_word in user_input.lower() for emotion_word in ['sad', 'angry', 'depressed', 'anxious', 'low']):
                print("User input indicates emotion (sad, angry, depressed, anxious, low)")

                # Simulate a delay for the pop-ups to show up (replace with actual processing time)
                time.sleep(2)

                # Generate and return messages
                messages = []

                for _ in range(5):
                    if cute_messages:
                        messages.append({'type': 'popup', 'content': cute_messages.pop(0)})
                    if jokes:
                        messages.append({'type': 'popup', 'content': jokes.pop(0)})

                print(f"Sending messages: {messages}")
                return jsonify({'messages': messages, 'prompt': 'Are you feeling better? If yes, yay! Keep smiling. If no, let me cheer you up more.', 'show_options': True})
            else:
                ai_response = generate_ai_response(user_input)
                print(f"Generated AI response: {ai_response}")
                return jsonify({'ai_response': ai_response, 'show_options': False})
        except Exception as e:
            print(f"Error during model inference: {e}")
            return jsonify({'ai_response': "Error generating AI response.", 'show_options': False})

@app.route('/handle_user_response', methods=['POST'])
def handle_user_response():
        try:
            user_response = request.json.get('user_response')

            print(f"Received user response: {user_response}")

            if user_response.lower() == 'no':
                print("User responded with 'no'")
                cute_messages.extend([
                    "i ap-peach-iate you 🧸",
                    "don't be sadish have a radish 🤤",
                    "I carrot alot about you 🥕",
                    "it's okay to put yourself first 😘",
                    "BREAKING NEWS! You're super cute 💞"
                ])

                jokes.extend([
                    "what do you call a bear with no ears? B(heheheh)",
                    "What do you call an ant that wont go away? 🐜 permanant",
                    "Why did the maths book look so sad?😔 it was full of problems🤭",
                    "What do you call a man with no body and just a nose👃 nobody nose🕺",
                    "How do you make seven an even number?🤔 take the s out(heheheh)"
                ])

                messages = []

                for _ in range(5):
                    if cute_messages:
                        messages.append({'type': 'popup', 'content': cute_messages.pop(0)})
                    if jokes:
                        messages.append({'type': 'popup', 'content': jokes.pop(0)})

                print(f"Sending messages: {messages}")
                return jsonify({'messages': messages, 'prompt': 'Are you feeling better? If yes, yay! Keep smiling. If no, let me cheer you up more.', 'show_options': True})
            elif user_response.lower() == 'yes':
                print("User responded with 'yes'")
                return jsonify({'ai_response': "Yayyy! You look beautiful in that smile!", 'show_options': False})
            else:
                print("User responded with an unexpected answer")
                return jsonify({'ai_response': "I'm here to chat! Feel free to share your thoughts.", 'show_options': False})
        except Exception as e:
            print(f"Error during model inference: {e}")
            return jsonify({'ai_response': "Error generating AI response.", 'show_options': False})

def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

def generate_ai_response(user_input):
        try:
            user_input_lower = user_input.lower()

            best_match = max(dataset, key=lambda example: similar(user_input_lower, example['user'].lower()))

            if similar(user_input_lower, best_match['user'].lower()) > 0.7:
                return best_match['assistant']

            # Placeholder for GPT-2 model response
            ai_response = "Placeholder response from GPT-2 model"

            return ai_response
        except Exception as e:
            print(f"Error during model inference: {e}")
            return "Error generating AI response."

if __name__ == '__main__':
    # Start the emotion detection thread
    # emotion_detection_thread.start()

    # Start the Flask application with SocketIO support
    # socketio.run(app, debug=True)
    # Run the flask app
    app.run()

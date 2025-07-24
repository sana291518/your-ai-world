Your AI World

Project Overview
"Your AI World" (originally "SentimentAI Hub") is an innovative web application designed to enhance user engagement through AI-driven chat interactions and real-time emotion recognition. This platform aims to revolutionize human-computer interaction by providing a user-friendly interface where individuals can interact with an AI chatbot and experience real-time emotion detection using their device's camera. It represents a paradigm shift towards more empathetic and personalized digital ecosystems, leveraging sophisticated AI algorithms and cutting-edge computer vision techniques to understand and adapt to users' emotional states.

Features
"Your AI World" offers advanced functionalities that redefine user interaction:

AI Chatbot: An intelligent chatbot trained on a diverse dataset, capable of understanding and responding intelligently to user queries in a conversational manner.

Real-time Emotion Recognition: Integrates technology to detect user emotions in real-time via the device's camera, allowing the system to adapt its responses based on detected emotional cues.

User-Centric Interface: Designed for intuitive navigation and seamless interaction, ensuring a dynamic and immersive user experience.

Personalized Interactions: Adapts responses based on detected emotions, fostering deeper connections and enhancing overall user engagement and personalization.

Robust Technological Infrastructure: Built on powerful libraries to perform complex machine learning and computer vision tasks, including emotion classification and facial expression analysis.

Technologies Used
"Your AI World" is built with a robust technological stack, primarily focusing on Python for backend logic and machine learning, alongside standard web development languages for the frontend:

Backend:

Python (version 3.x): The primary programming language for the web application and machine learning algorithms.

Flask: A lightweight Python web framework serving as the backbone for routing and communication between frontend and backend.

TensorFlow: An open-source machine learning framework used for building and training machine learning models, including emotion recognition.

Keras: A high-level neural networks API, running on top of TensorFlow, for model development.

OpenCV: An open-source computer vision library for image processing and facial expression recognition.

Frontend:

HTML: For structuring the web pages and content.

CSS: For styling and ensuring an aesthetically pleasing and responsive design.

JavaScript: For interactive elements and enhancing the user experience.

Development Tools:

Web Browser: Any modern web browser (Google Chrome, Mozilla Firefox, Safari) for accessing and testing.

Git: For source code management and version control.

Installation and Setup
To set up and run "Your AI World" locally, follow these steps:

Clone the repository:

git clone https://github.com/your-username/your-ai-world.git
cd your-ai-world

Install Python dependencies:
This project requires Python and several machine learning libraries. It is recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv
# Activate the virtual environment
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
# Install required packages
pip install Flask TensorFlow Keras opencv-python

Note: Specific versions of these libraries might be required based on your model. Refer to a requirements.txt if available in the project files.

Model Setup (if applicable):
Ensure that any pre-trained models for emotion recognition or the chatbot are correctly placed in the project directory as expected by the Flask application. (Details on model file paths would typically be in the project's internal documentation).

Usage
To run and interact with the "Your AI World" application:

Start the Flask application:
Ensure your virtual environment is activated (as per installation steps).

python app.py
# (Assuming your main Flask application file is named app.py)

The application will typically start on http://127.0.0.1:5000/ or a similar local address.

Access the application:
Open your web browser and navigate to the address provided in your terminal after starting the Flask app.

Interact with the AI:

Engage with the AI chatbot through the chat interface.

If a webcam is connected, allow camera access to utilize the real-time emotion recognition feature.

Contributing
This project was developed as an academic mini-project. While direct contributions might not be actively sought for this specific repository, feedback, suggestions, and ideas for future enhancements are always welcome.

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Commit your changes (git commit -m 'Add your feature description').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

License
This project is licensed under the MIT License.

Made by Sana Mapkar, Jui Magar, and Shrushti Ramnathkar

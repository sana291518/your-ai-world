function showPopup() {
    document.getElementById('popup').style.display = 'block';
}

function startAI() {
    document.getElementById('chatBox').style.display = 'block';
    closePopup(); // Close the popup after starting AI
}

function closePopup() {
    document.getElementById('popup').style.display = 'none';
}

function sendMessage() {
    // Get user input
    var userInput = document.getElementById('userInput').value;

    // Add user message to the chat
    addToChat('user', userInput);

    // Send the user input to your server for processing
    fetch('/process_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'user_input': userInput,
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Add AI response to the chat
        addToChat('ai', data.ai_response);

        // Check if user input indicates sadness or anger
        if (data.show_cute_message) {
            // Show a cute message
            getCuteMessage();
        } else if (data.show_joke) {
            // Show a joke
            getJoke();
        } else if (data.show_options) {
            // Display the modal directly
            showModal(data.prompt);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Clear the input field
    document.getElementById('userInput').value = '';
}

function addToChat(sender, message) {
    var chat = document.getElementById('chat');
    var newMessage = document.createElement('div');
    newMessage.className = sender + 'Bubble'; // Apply userBubble or aiBubble class
    newMessage.innerText = message;
    chat.appendChild(newMessage);
}

function getCuteMessage() {
    fetch('/get_cute_message', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        showModal(data.cute_message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function getJoke() {
    fetch('/get_joke', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        showModal(data.joke);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function showModal(message) {
    // Get the modal element
    var modal = document.getElementById("myModal");

    // Check if the modal element exists
    if (modal) {
        // Display the modal
        modal.style.display = "block";

        // Display the message in the modal content
        var modalContent = document.getElementById("modal-content");
        modalContent.innerHTML = message;
    } else {
        console.error("Modal element not found");
    }
}

function closeModal() {
    var modal = document.getElementById("myModal");
    if (modal) {
        modal.style.display = "none";
    } else {
        console.error("Modal element not found");
    }
}

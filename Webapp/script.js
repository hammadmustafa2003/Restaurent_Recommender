document.addEventListener('DOMContentLoaded', function () {
  function addMessage(sender, content) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.innerHTML = content;
    chatBox.appendChild(messageDiv);
  }

  // Function to handle user input
  async function handleUserInput() {
    const inputField = document.querySelector('input[type="text"]');
    const message = inputField.value.trim();

    if (message !== '') {
      addMessage('sent', message);
      inputField.value = '';

      // Make API call to get the next entity
      const response = await fetch('http://127.0.0.1:8000/get_entity/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sentence: message, entity: currentEntity }),
      });

      const data = await response.json();
      addMessage('received', data.next_message);
      if (currentEntity != data.next_entity && data.predicted_entity != null && data.next_entity != null){
        switch (currentEntity) {
          case 'Location':
            currentLocation = data.predicted_entity;
            break;
          case 'Cuisine':
            currentCuisine = data.predicted_entity;
            break;
          case 'Price':
            currentPrice = parseInt(data.predicted_entity);
            break;
        }
        currentEntity = data.next_entity;
      }

      else if (data.next_entity == null){
        // Make API call to get the recommendation
        const response = await fetch('http://127.0.0.1:8000/recommend/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ location: currentLocation, cuisine: currentCuisine, budget: currentPrice }),
        });
  
        const data = await response.json();
        addMessage('received', data.recommendation+"\n\nThank you for using our service. Have a nice day! Please enter the location again to continue.");
        currentEntity = 'Location';
        currentLocation = '';
        currentCuisine = '';
        currentPrice = '';
      }
      
    }
  }

  // Send button event listener
  const sendButton = document.getElementById('send-button');
  sendButton.addEventListener('click', handleUserInput);

  // Input field event listener for pressing Enter key
  const inputField = document.querySelector('input[type="text"]');
  inputField.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
      handleUserInput();
    }
  });

  // Function to scroll the chat to the latest message
  function scrollToBottom() {
    const chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Scroll to the bottom initially
  scrollToBottom();

  // Global variable to keep track of the current entity in the conversation
  let currentEntity = 'Location';
  let currentLocation = '';
  let currentCuisine = '';
  let currentPrice = 0;

  // Initial welcome message
  addMessage('received', 'Welcome to our restaurant recommendation system. Could you please tell me your location?');
});
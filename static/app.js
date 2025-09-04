const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

// ✅ Connect WebSocket
const ws = new WebSocket(`ws://${window.location.host}/ws?session_id=user1`);

ws.onopen = () => {
    console.log("✅ Connected to WebSocket");
};

ws.onclose = () => {
    console.log("❌ WebSocket disconnected");
};

ws.onmessage = (event) => {
    displayMessage(event.data, "bot");
};

// ✅ Display Messages
function displayMessage(message, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
    msgDiv.innerText = message;
    chatBox.appendChild(msgDiv);

    // Auto scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ✅ Send Messages
chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    displayMessage(message, "user");
    ws.send(message);
    userInput.value = "";
});

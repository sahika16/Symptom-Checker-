let lastPrediction = null;
let awaitingRemedy = false;
let selectedLanguage = "en";
let remediesShown = false;
let isTranslating = false;

function speak(text, lang) {
    if (!text) return;
    
    window.speechSynthesis.cancel();
    
    var speech = new SpeechSynthesisUtterance(text);
    
    var langMap = {
        "mr": "hi-IN",
        "hi": "hi-IN",
        "en": "en-US"
    };
    
    speech.lang = langMap[lang] || "en-US";
    speech.rate = 0.9;
    speech.pitch = 1;
    speech.volume = 1;
    
    var voices = window.speechSynthesis.getVoices();
    var voice = null;
    
    if (lang === "mr" || lang === "hi") {
        voice = voices.find(function(v) { return v.lang.includes("IN"); });
    } else {
        voice = voices.find(function(v) { return v.lang.startsWith("en"); });
    }
    
    if (voice) {
        speech.voice = voice;
    }
    
    window.speechSynthesis.speak(speech);
}

function handleKeyPress(e) {
    if (e.key === "Enter") sendMessage();
}

function appendMessage(role, text) {
    var chatBox = document.getElementById("chatBox");
    var div = document.createElement("div");
    div.className = "message " + role;
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
}

function switchLanguage() {
    if (isTranslating) return;
    
    var newLang = document.getElementById("languageSelect").value;
    selectedLanguage = newLang;
    
    var welcome = document.getElementById("welcomeMessage");
    if (newLang === "mr") {
        welcome.innerText = "स्वागत आहे. कृपया तुमच्या लक्षणांचे वर्णन करा.";
    } else if (newLang === "hi") {
        welcome.innerText = "स्वागत है। कृपया अपने लक्षणों का वर्णन करें।";
    } else {
        welcome.innerText = "Welcome. Please describe your symptoms.";
    }
    
    var botMessages = document.querySelectorAll(".message.bot");
    var botTexts = [];
    
    botMessages.forEach(function(msg) {
        botTexts.push(msg.innerText);
    });
    
    if (botTexts.length === 0) return;
    
    isTranslating = true;
    
    fetch("http://127.0.0.1:5000/translate-all", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            texts: botTexts,
            language: newLang
        })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        if (data.translated_texts) {
            var translated = data.translated_texts;
            for (var i = 0; i < botMessages.length && i < translated.length; i++) {
                botMessages[i].innerText = translated[i];
            }
            
            if (translated.length > 0) {
                speak(translated[translated.length - 1], newLang);
            }
        }
        isTranslating = false;
    })
    .catch(function() {
        isTranslating = false;
    });
}

function sendMessage() {
    var input = document.getElementById("userInput");
    var message = input.value.trim();
    
    if (!message) return;
    
    appendMessage("user", message);
    input.value = "";
    
    if (awaitingRemedy && !remediesShown) {
        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                symptoms: message,
                language: selectedLanguage,
                is_remedy_response: true
            })
        })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.message === "remedy_request") {
                fetchRemedy();
                return;
            }
            appendMessage("bot", data.message);
            speak(data.message, selectedLanguage);
            awaitingRemedy = false;
            lastPrediction = null;
            remediesShown = false;
        });
        return;
    }
    
    awaitingRemedy = false;
    remediesShown = false;
    lastPrediction = null;
    
    var loadingText = selectedLanguage === "mr" ? "लक्षणांचे विश्लेषण सुरू आहे..." :
                      selectedLanguage === "hi" ? "लक्षणों का विश्लेषण किया जा रहा है..." :
                      "Analyzing symptoms...";
    
    var botMsg = appendMessage("bot", loadingText);
    
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            symptoms: message,
            language: selectedLanguage,
            is_remedy_response: false
        })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        if (data.is_general) {
            botMsg.innerText = data.message;
            speak(data.message, selectedLanguage);
            lastPrediction = null;
            awaitingRemedy = false;
            remediesShown = false;
            return;
        }
        
        if (!data.prediction) {
            botMsg.innerText = data.message || "Please describe your symptoms.";
            speak(botMsg.innerText, selectedLanguage);
            lastPrediction = null;
            awaitingRemedy = false;
            remediesShown = false;
            return;
        }
        
        lastPrediction = data.prediction;
        awaitingRemedy = true;
        remediesShown = false;
        
        var responseText = data.you_may_have + "\n\n" + data.would_you_like;
        
        botMsg.innerText = responseText;
        speak(responseText, selectedLanguage);
    })
    .catch(function() {
        var msg = selectedLanguage === "mr" ? "सर्व्हर त्रुटी. कृपया पुन्हा प्रयत्न करा." :
                  selectedLanguage === "hi" ? "सर्वर त्रुटि। कृपया फिर से प्रयास करें।" :
                  "Server error. Please try again.";
        botMsg.innerText = msg;
        speak(msg, selectedLanguage);
        lastPrediction = null;
        awaitingRemedy = false;
        remediesShown = false;
    });
}

function fetchRemedy() {
    if (!lastPrediction) return;
    
    var loadingText = selectedLanguage === "mr" ? "उपाय मिळवत आहे..." :
                      selectedLanguage === "hi" ? "सुझाव प्राप्त किए जा रहे हैं..." :
                      "Fetching suggestions...";
    
    var botMsg = appendMessage("bot", loadingText);
    
    fetch("http://127.0.0.1:5000/remedy", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            type: lastPrediction,
            language: selectedLanguage
        })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        var responseText = "";
        if (selectedLanguage === "mr") {
            responseText = data.condition_label + ": " + lastPrediction + "\n\n" + data.remedy_label + ":\n" + data.remedy + "\n\n" + data.precaution_label + ":\n" + data.precaution + "\n\n" + data.doctor_label + ":\n" + data.doctor_advice;
        } else if (selectedLanguage === "hi") {
            responseText = data.condition_label + ": " + lastPrediction + "\n\n" + data.remedy_label + ":\n" + data.remedy + "\n\n" + data.precaution_label + ":\n" + data.precaution + "\n\n" + data.doctor_label + ":\n" + data.doctor_advice;
        } else {
            responseText = data.condition_label + ": " + lastPrediction + "\n\n" + data.remedy_label + ":\n" + data.remedy + "\n\n" + data.precaution_label + ":\n" + data.precaution + "\n\n" + data.doctor_label + ":\n" + data.doctor_advice;
        }
        
        botMsg.innerText = responseText;
        speak(responseText, selectedLanguage);
        awaitingRemedy = false;
        lastPrediction = null;
        remediesShown = true;
    })
    .catch(function() {
        var msg = selectedLanguage === "mr" ? "उपाय मिळवताना त्रुटी." :
                  selectedLanguage === "hi" ? "सुझाव प्राप्त करने में त्रुटि।" :
                  "Error fetching remedies.";
        botMsg.innerText = msg;
        awaitingRemedy = false;
        lastPrediction = null;
        remediesShown = true;
    });
}

// Voice recording
var mediaRecorder = null;
var audioChunks = [];

function toggleRecording() {
    var btn = document.querySelector(".mic-btn");
    
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.ondataavailable = function(e) { audioChunks.push(e.data); };
            mediaRecorder.onstop = sendAudio;
            mediaRecorder.start();
            
            btn.innerHTML = "Stop";
            btn.style.backgroundColor = "#ff4444";
        })
        .catch(function() { alert("Microphone access denied."); });
    } else {
        mediaRecorder.stop();
        btn.innerHTML = "Speak";
        btn.style.backgroundColor = "";
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(function(t) { t.stop(); });
        }
    }
}

function sendAudio() {
    var blob = new Blob(audioChunks, { type: "audio/webm" });
    var formData = new FormData();
    formData.append("audio", blob, "recording.webm");
    formData.append("language", selectedLanguage);
    
    var loadingText = selectedLanguage === "mr" ? "व्हॉइस ट्रान्सक्राइब होत आहे..." :
                      selectedLanguage === "hi" ? "वॉइस ट्रांसक्राइब हो रहा है..." :
                      "Transcribing voice...";
    
    var botMsg = appendMessage("bot", loadingText);
    
    fetch("http://127.0.0.1:5000/transcribe", {
        method: "POST",
        body: formData
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        if (data.transcript) {
            document.getElementById("userInput").value = data.transcript;
            botMsg.remove();
            sendMessage();
        } else {
            botMsg.innerText = "Could not transcribe.";
        }
    });
}

function toggleDarkMode() {
    document.body.classList.toggle("dark");
    if (document.body.classList.contains("dark")) {
        localStorage.setItem("theme", "dark");
    } else {
        localStorage.setItem("theme", "light");
    }
}

window.addEventListener("load", function() {
    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark");
        var toggle = document.querySelector(".theme-toggle input");
        if (toggle) toggle.checked = true;
    }
});

document.getElementById("languageSelect").addEventListener("change", function() {
    switchLanguage();
});
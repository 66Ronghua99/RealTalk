let ws = null;
let accumulatedCount = 0;

// Audio chunk accumulation
let audioChunks = [];
let isPlayingAudio = false;  // P0 fix: track playback state to prevent duplicates

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        addSystemMessage('Connected to RealTalk');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        addSystemMessage('Disconnected');
        setTimeout(connect, 3000);
    };
}

function handleMessage(data) {
    switch(data.type) {
        case 'transcript':
            addMessage(data.text, 'user');
            break;
        case 'llm_chunk':
            // Stream LLM response to display (update last message)
            const chat = document.getElementById('chat');
            const messages = chat.querySelectorAll('.message.assistant');
            if (messages.length > 0) {
                messages[messages.length - 1].textContent = data.text;
            } else {
                addMessage(data.text, 'assistant');
            }
            break;
        case 'tts_audio':
            // Accumulate audio chunks and play when complete
            if (data.audio) {
                console.log('Received tts_audio, length:', data.audio.length, 'is_final:', data.is_final);

                // P0 fix: Reset audio chunks if we're starting a new TTS stream
                // This prevents race condition where old chunks persist
                if (isPlayingAudio) {
                    console.log('New TTS while audio playing - resetting chunks');
                    audioChunks = [];
                    isPlayingAudio = false;
                }

                // Decode base64 to binary
                const binaryString = atob(data.audio);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                audioChunks.push(bytes);

                // Only play when all chunks are received
                if (data.is_final) {
                    console.log('All chunks received, total:', audioChunks.length);
                    addSystemMessage('Playing TTS audio...');

                    try {
                        // Combine all chunks
                        const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
                        const combined = new Uint8Array(totalLength);
                        let offset = 0;
                        for (const chunk of audioChunks) {
                            combined.set(chunk, offset);
                            offset += chunk.length;
                        }

                        // Debug: show first bytes
                        let hexDump = '';
                        for (let i = 0; i < Math.min(16, combined.length); i++) {
                            hexDump += combined[i].toString(16).padStart(2, '0') + ' ';
                        }
                        console.log('First bytes (hex):', hexDump);

                        // Play combined MP3
                        const blob = new Blob([combined], { type: 'audio/mpeg' });
                        const audioUrl = URL.createObjectURL(blob);
                        console.log('Creating Audio element, url:', audioUrl);
                        const audio = new Audio(audioUrl);

                        // P0 fix: Set playback flag before playing
                        isPlayingAudio = true;

                        // Prevent duplicate play
                        let played = false;
                        const doPlay = () => {
                            if (played) {
                                console.log('Audio already played, skipping');
                                return;
                            }
                            played = true;
                            console.log('Calling audio.play()');
                            return audio.play();
                        };

                        doPlay().then(() => {
                            console.log('Audio playing, duration:', audio.duration);
                        }).catch(err => {
                            console.error('Audio play error:', err);
                            isPlayingAudio = false;  // Reset flag on error
                        });

                        audio.onended = () => {
                            console.log('Audio playback finished');
                            URL.revokeObjectURL(audioUrl);
                            audioChunks = []; // Clear for next time
                            isPlayingAudio = false;  // P0 fix: Reset playback flag
                        };
                    } catch (e) {
                        console.error('TTS audio error:', e);
                        addSystemMessage('TTS error: ' + e.message);
                        audioChunks = [];
                        isPlayingAudio = false;  // P0 fix: Reset playback flag on error
                    }
                }
            }
            break;
        case 'state':
            document.getElementById('state').textContent = data.state.toUpperCase();
            if (data.state === 'speaking') {
                addMessage(data.text || 'AI is speaking...', 'assistant');
            }
            break;
        case 'gatekeeper':
            document.getElementById('gatekeeper').textContent = data.action.toUpperCase();
            break;
        case 'status':
            addSystemMessage(data.message);
            if (data.message.includes('Accumulated')) {
                accumulatedCount++;
                document.getElementById('accumulated').textContent = accumulatedCount;
            }
            break;
        case 'vad':
            // Update VAD status indicator
            const micBtn = document.getElementById('micBtn');
            if (data.is_speaking) {
                micBtn.style.background = '#00ff00';
            } else {
                micBtn.style.background = '#ff4757';
            }
            break;
    }
}

function sendMessage() {
    const input = document.getElementById('message');
    const text = input.value.trim();
    if (!text || !ws) return;

    ws.send(JSON.stringify({type: 'text', text}));
    input.value = '';
}

function interrupt() {
    if (!ws) return;
    ws.send(JSON.stringify({type: 'interrupt', text: 'User interrupted'}));
}

function clearContext() {
    if (!ws) return;
    ws.send(JSON.stringify({type: 'clear'}));
    accumulatedCount = 0;
    document.getElementById('accumulated').textContent = '0';
}

// Test audio playback using Web Audio API
function testAudio() {
    console.log('Testing audio playback...');
    addSystemMessage('Testing audio playback...');

    try {
        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Generate a simple 440Hz sine wave (A note)
        const sampleRate = audioContext.sampleRate;
        const duration = 1.0; // 1 second
        const frequency = 440;
        const numSamples = sampleRate * duration;
        const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
        const channelData = audioBuffer.getChannelData(0);

        // Generate sine wave with fade in/out
        for (let i = 0; i < numSamples; i++) {
            const t = i / sampleRate;
            // Apply envelope (fade in/out)
            let envelope = 1;
            if (i < sampleRate * 0.1) {
                envelope = i / (sampleRate * 0.1);
            } else if (i > numSamples - sampleRate * 0.1) {
                envelope = (numSamples - i) / (sampleRate * 0.1);
            }
            channelData[i] = Math.sin(2 * Math.PI * frequency * t) * 0.5 * envelope;
        }

        // Play the audio
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);

        source.onended = () => {
            console.log('Test audio played successfully!');
            addSystemMessage('Test audio played! Did you hear the sound?');
        };

        source.start(0);
        console.log('Test audio started');
    } catch (e) {
        console.error('Test audio error:', e);
        addSystemMessage('Test audio error: ' + e.message);
    }
}

function handleKeyPress(e) {
    if (e.key === 'Enter') sendMessage();
}

function addMessage(text, role) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function addSystemMessage(text) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'message system';
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// Stubbornness slider
document.getElementById('stubbornness').addEventListener('input', (e) => {
    document.getElementById('stubbornness-val').textContent = e.target.value;
    if (ws) {
        ws.send(JSON.stringify({type: 'stubbornness', level: parseInt(e.target.value)}));
    }
});

// Microphone audio capture with frontend VAD (like Open-LLM-VTuber)
let isRecording = false;
let audioContext = null;
let micStream = null;
let processor = null;

// VAD state
let isSpeaking = false;
let silenceCount = 0;
let audioBuffer = [];
const SILENCE_THRESHOLD = 0.01;
const SILENCE_FRAMESneeded = 5;  // ~1.3s at 5 frames

async function toggleMic() {
    const btn = document.getElementById('micBtn');

    if (!isRecording) {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }});

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            await audioContext.resume();

            const source = audioContext.createMediaStreamSource(micStream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer;
                const channelData = inputBuffer.getChannelData(0);

                // Frontend VAD: calculate energy
                let sum = 0;
                for (let i = 0; i < channelData.length; i++) {
                    sum += channelData[i] * channelData[i];
                }
                const rms = Math.sqrt(sum / channelData.length);

                // DEBUG: Log WS state
                const wsState = ws ? ws.readyState : 'null';
                // console.log('onaudioprocess: ws.readyState=', wsState, 'rms=', rms.toFixed(4));

                // Send audio data
                const audioArray = Array.from(channelData);
                if (ws && ws.readyState === WebSocket.OPEN) {
                    // console.log('Sending mic-audio-data, length=', audioArray.length);
                    ws.send(JSON.stringify({
                        type: 'mic-audio-data',
                        audio: audioArray
                    }));
                } else {
                    console.warn('WebSocket not open, state:', wsState);
                }

                // VAD logic
                if (rms > SILENCE_THRESHOLD) {
                    if (!isSpeaking) {
                        isSpeaking = true;
                        console.log('Speech started');
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({type: 'audio_start'}));
                        }
                    }
                    silenceCount = 0;
                } else {
                    if (isSpeaking) {
                        silenceCount++;
                        if (silenceCount > SILENCE_FRAMESneeded) {
                            // Speech ended
                            isSpeaking = false;
                            silenceCount = 0;
                            console.log('Speech ended');
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({type: 'mic-audio-end'}));
                            }
                        }
                    }
                }
            };

            // Connect processor to enable callback (with gain=0 to prevent echo)
            source.connect(processor);
            const dummy = audioContext.createGain();
            dummy.gain.value = 0;
            processor.connect(dummy);
            dummy.connect(audioContext.destination);

            isRecording = true;
            btn.classList.add('recording');
            btn.textContent = '⏹';
            btn.style.background = '#00ff00';

            addSystemMessage('Recording started...');

        } catch (err) {
            addSystemMessage('Microphone error: ' + err.message);
            console.error('Mic error:', err);
        }
    } else {
        stopRecording();
    }
}

function stopRecording() {
    const btn = document.getElementById('micBtn');

    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    isRecording = false;
    isSpeaking = false;
    silenceCount = 0;

    btn.classList.remove('recording');
    btn.textContent = '🎤';
    btn.style.background = '#ff4757';

    addSystemMessage('Recording stopped');
}

connect();

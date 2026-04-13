from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
import numpy as np
from ultralytics import YOLO
from groq import Groq
from duckduckgo_search import DDGS  # for Live internet

app = Flask(__name__)

# ==========================================
# 1. SETUP GROQ AI (Fastest & Error-Free)
# ==========================================
# GitHub block se bachne ke liye hum key 'os.environ' se le rahe hain
# (Render automatically apni settings se isme key daal dega)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# 2. SETUP YOLO MODEL
# ==========================================
try:
    model_yolo = YOLO('best.pt') 
except Exception as e:
    print(f"Warning: YOLO Model not found. Error: {e}")
    model_yolo = None

# --- GLOBAL VARIABLES ---
sensitivity = 0.5
latest_detections = [] 

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_objects')
def get_objects():
    global latest_detections
    return jsonify({"objects": latest_detections})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global sensitivity
    data = request.json
    sensitivity = float(data.get('sensitivity', 0.5))
    return jsonify({"status": "success"})

# --- NAYA MOBILE CAMERA ROUTE (Phone se photo receive karne ke liye) ---
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    global sensitivity, latest_detections
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"objects": []})

        # Phone se aayi hui base64 photo ko padhna
        image_data = data['image'].split(',')[1] 
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        current_objects = []
        
        # YOLO Processing
        if model_yolo:
            results = model_yolo(frame, conf=sensitivity)
            frame_area = frame.shape[0] * frame.shape[1]
            
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                class_name = model_yolo.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                ratio = box_area / frame_area
                
                if ratio > 0.35: distance = "very close to you"
                elif ratio > 0.10: distance = "nearby"
                else: distance = "a bit far away"
                    
                current_objects.append(f"{class_name} that is {distance}")

        latest_detections = list(set(current_objects))
        return jsonify({"objects": latest_detections})
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"objects": []})

# --- AURA SENSE BRAIN (GROQ AI + LIVE INTERNET) ---
@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    data = request.json
    prompt = data.get('prompt', '')
    
    try:
        print(f"\n[AURA HEARD]: {prompt}") 
        
        # 1. LIVE INTERNET SEARCH
        live_context = ""
        try:
            print("[SYSTEM]: Fetching live internet data...")
            results = DDGS().text(prompt, max_results=2)
            if results:
                search_text = " ".join([res['body'] for res in results])
                live_context = f"\n\nHere is real-time internet data to help you answer accurately: {search_text}"
        except Exception:
            print("[SYSTEM]: Internet search skipped.")

        # 2. GROQ AI CALL 
        system_instruction = (
            "You are Aura Sense, an AI assistant built to help a blind person navigate the world. "
            "Answer the user's query clearly, conversationally, and keep it very brief (1-2 sentences maximum)."
            + live_context
        )

        if not client:
            return jsonify({"response": "API Key is missing. Please check Render Environment Variables."})

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.4,
        )
        
        raw_text = chat_completion.choices[0].message.content
        clean_text = raw_text.replace('*', '') 
        
        print(f"[AURA ANSWERED]: {clean_text}\n") 
        return jsonify({"response": clean_text})
    
    except Exception as e:
        print(f"\n❌ GROQ ERROR: {e}\n")
        return jsonify({"response": "I'm sorry, I am having trouble connecting to my brain right now."})

if __name__ == '__main__':
    # Render cloud ke hisaab se port setting
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
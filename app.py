from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
from ultralytics import YOLO
from groq import Groq
from duckduckgo_search import DDGS  # for Live internet

app = Flask(__name__)

# ==========================================
# 1. SETUP GROQ AI (Fastest & Error-Free)
# ==========================================
# Groq API key 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

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
process_frames = 10
latest_detections = [] 
camera = cv2.VideoCapture(0)

# --- CAMERA & YOLO LOGIC ---
def generate_frames():
    global latest_detections
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # flip to remove Mirror effect
        frame = cv2.flip(frame, 1)
        frame_count += 1
            
        if model_yolo and frame_count % process_frames == 0:
            results = model_yolo(frame, conf=sensitivity)
            frame = results[0].plot()
            
            current_objects = []
            frame_area = frame.shape[0] * frame.shape[1]
            
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                class_name = model_yolo.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                ratio = box_area / frame_area
                
                if ratio > 0.35:
                    distance = "very close to you"
                elif ratio > 0.10:
                    distance = "nearby"
                else:
                    distance = "a bit far away"
                    
                current_objects.append(f"{class_name} that is {distance}")
            
            latest_detections = list(set(current_objects))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_objects')
def get_objects():
    global latest_detections
    return jsonify({"objects": latest_detections})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global sensitivity, process_frames
    data = request.json
    sensitivity = float(data.get('sensitivity', 0.5))
    process_frames = int(data.get('frames', 10))
    return jsonify({"status": "success"})

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
            # getting current news from DuckDuckGo 
            results = DDGS().text(prompt, max_results=2)
            if results:
                search_text = " ".join([res['body'] for res in results])
                live_context = f"\n\nHere is real-time internet data to help you answer accurately: {search_text}"
        except Exception as e:
            print("[SYSTEM]: Internet search skipped.")

        # 2. GROQ AI CALL (Fast Llama model with Internet Data)
        system_instruction = (
            "You are Aura Sense, an AI assistant built to help a blind person navigate the world. "
            "Answer the user's query clearly, conversationally, and keep it very brief (1-2 sentences maximum)."
            + live_context
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", # Groq's superfast model
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
    app.run(debug=True, port=5000, threaded=True)
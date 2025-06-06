import cv2
import face_recognition
import re
import pyttsx3
import tkinter as tk
import cloudinary
import cloudinary.uploader
from tkinter import filedialog, messagebox
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient
import time
import numpy as np

# ----- Configuration -----
QDRANT_URL = "<URL>"  # Update with actual URL
API_KEY = "<API_key>"  # Replace with your API key
MONGO_URI = ""  # MongoDB connection URI
DB_NAME = "FaceDB"
COLLECTION_NAME = "face_attendance"
VECTOR_DIM = 128  # face_recognition embeddings

# ----- Initialize Clients -----
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
users_collection = db["users"]

# Cloudinary Configuration
cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret="",
    secure=True
)

# Initialize TTS engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Ensure Qdrant collection exists
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

# Function to upload face image to Cloudinary
def upload_to_cloudinary(image_path):
    upload_result = cloudinary.uploader.upload(image_path)
    return upload_result["secure_url"]

# ----- Input Validation -----
def validate_inputs(name, user_id, prn_no):
    if not name.strip():
        messagebox.showerror("Error", "Name cannot be empty")
        speak("Name cannot be empty")
        return False
    if not user_id.isdigit():
        messagebox.showerror("Error", "User ID must be a number")
        speak("User ID must be a number")
        return False
    if not re.match(r"^[0-9]{3}$", prn_no):
        messagebox.showerror("Error", "PRN No must be a 3-digit number")
        speak("PRN Number must be a 3-digit number")
        return False
    return True

# ----- Utility Functions -----
def get_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise ValueError("No face detected")
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)
    
    # Extract face region
    top, right, bottom, left = face_locations[0]  # Assuming one face per image
    face_image = image[top:bottom, left:right]
    
    # Save cropped face image
    face_image_path = image_path.replace("temp_", "face_")
    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    
    return face_encoding[0], face_image_path, face_image  # Return embedding and cropped face path

def capture_photo():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            speak("Failed to capture photo")
            messagebox.showerror("Error", "Failed to capture photo")
            cap.release()
            return None
        countdown = 3 - int(time.time() - start_time)
        # speak(f"{countdown}")
        cv2.putText(frame, f"Capturing in {countdown}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    file_path = "captured_face.jpg"
    cv2.imwrite(file_path, frame)
    speak("Photo captured successfully")
    return file_path

def add_face(name, user_id, prn_no, image_path):
    if not validate_inputs(name, user_id, prn_no):
        return
    try:
        embedding, face_image_path, face_image = get_face_embedding(image_path)
        
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        speak(str(e))
        return
    
    # Store in Qdrant
    point_id = int(user_id)
    insert_result = qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=embedding.tolist())]
    )
    if not insert_result:
        speak("Failed to add face to the database")
        messagebox.showerror("Error", "Failed to add face to the database")
        return
    
    speak(f"{name}'s, Face added to the Vector DB!")
    print("Face added successfully to the Vector DB!")


    # Upload to Cloudinary
    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    cloudinary_url = upload_to_cloudinary(face_image_path)
    speak(f"{name}'s, Face added to the Cloudinary!")
    print("Face added to the Cloudinary!")

    # Store in MongoDB
    users_collection.insert_one({"_id": point_id, "name": name, "prn_no": prn_no, "face_image_url": cloudinary_url})
    speak(f"{name}'s, Face added to the database!")
    print("Face added successfully to the database!")
    messagebox.showinfo("Success", "Face added successfully!")

def recognize_face(image_path):
    try:
        query_embedding, _, _ = get_face_embedding(image_path)
    except ValueError as e:
        speak(str(e))
        messagebox.showerror("Error", str(e))
        return
    speak("Recognizing face, Please wait...")
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_embedding.tolist(), limit=1
    )
    
    if not search_results.points:
        print(search_results.points)
        speak("No matching face found")
        messagebox.showinfo("Result", "No matching face found")
        return
    
    best_match = search_results.points[0]

    if best_match.score > 0.97:
        user_data = users_collection.find_one({"_id": best_match.id})
        speak(f"Welcome {user_data['name']}")
        messagebox.showinfo("Result", f"User ID: {best_match.id}\nName: {user_data['name']}\nPRN No: {user_data['prn_no']}")
    else:
        messagebox.showinfo("Result", "No matching face found")
        

# ----- UI Setup -----
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("600x600")
root.configure(bg="#34495E")

font_style = ("Arial", 14, "bold")
button_style = {"font": ("Arial", 12, "bold"), "bg": "#1ABC9C", "fg": "white", "bd": 3, "relief": "raised"}

def create_label(text):
    return tk.Label(root, text=text, font=font_style, bg="#34495E", fg="white")

def create_entry():
    return tk.Entry(root, font=("Arial", 12), bg="white", fg="black", bd=2, relief="solid")

# Loading Animation
loading_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#34495E", fg="yellow")
loading_label.pack(pady=5)

def show_loading(message):
    loading_label.config(text=message)
    root.update_idletasks()

def hide_loading():
    loading_label.config(text="")

# ----- Input Fields -----
create_label("Name:").pack(pady=5)
name_entry = create_entry()
name_entry.pack(pady=5)

create_label("User ID:").pack(pady=5)
user_id_entry = create_entry()
user_id_entry.pack(pady=5)

create_label("PRN No:").pack(pady=5)
prn_entry = create_entry()
prn_entry.pack(pady=5)

# ----- Button Functions -----
def upload_and_add(is_file_path=False, file_path=None):
    if not is_file_path:
        file_path = filedialog.askopenfilename()
        name = name_entry.get()
        user_id = user_id_entry.get()
        prn_no = prn_entry.get()
        speak(f"Adding {name}'s face, Please wait...")
        show_loading("Adding Face... Please Wait...")
        add_face(name, user_id, prn_no, file_path)
        hide_loading()
    else:
        name = name_entry.get()
        user_id = user_id_entry.get()
        prn_no = prn_entry.get()
        speak(f"Adding {name}'s face, Please wait...")
        show_loading("Adding Face... Please Wait...")
        add_face(name, user_id, prn_no, file_path)
        hide_loading()

def upload_and_recognize():
    file_path = filedialog.askopenfilename()
    if file_path:
        show_loading("Recognizing Face... Please Wait...")
        recognize_face(file_path)
        hide_loading()

def capture_and_add():
    speak("Capturing photo, Please wait...")
    show_loading("Capturing Photo... Please Wait...")
    file_path = capture_photo()
    hide_loading()
    if file_path:
        upload_and_add(file_path = file_path, is_file_path=True)

def capture_and_recognize():
    speak("Capturing photo, Please wait...")
    show_loading("Capturing Photo... Please Wait...")
    file_path = capture_photo()
    hide_loading()
    if file_path:
        recognize_face(file_path)

# ----- Buttons -----
tk.Button(root, text="Upload & Add Face", command=upload_and_add, **button_style).pack(pady=5)
tk.Button(root, text="Upload & Recognize Face", command=upload_and_recognize, **button_style).pack(pady=5)
tk.Button(root, text="Capture & Add Face", command=capture_and_add, **button_style).pack(pady=5)
tk.Button(root, text="Capture & Recognize Face", command=capture_and_recognize, **button_style).pack(pady=5)
tk.Button(root, text="Exit", command=root.quit, **button_style).pack(pady=10)

root.mainloop()
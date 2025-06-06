from fastapi import FastAPI, HTTPException, UploadFile, File
import cv2
import numpy as np
import face_recognition
import shutil
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient

# ----- Configuration -----
QDRANT_URL = "<URL>"  # Update with actual URL
API_KEY = "<API_key>"  # Replace with your API key
MONGO_URI = "<MONGO_URI>"  # MongoDB connection URI
DB_NAME = "FaceDB"
COLLECTION_NAME = "face_attendance"
VECTOR_DIM = 128  # face_recognition embeddings

# ----- Initialize Clients -----
app = FastAPI()
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
users_collection = db["users"]

# Ensure Qdrant collection exists
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

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
    
    return face_encoding[0], face_image_path  # Return embedding and cropped face path

# ----- API Routes -----
@app.post("/add_face")
async def add_face(name: str, user_id: str, prn_no: str, file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        embedding, face_image_path = get_face_embedding(file_path)
    except ValueError as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    os.remove(file_path)
    
    # Store in Qdrant
    point_id = int(user_id)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=embedding.tolist(), payload={"name": name})]
    )
    
    # Store in MongoDB
    users_collection.insert_one({"_id": point_id, "name": name, "prn_no": prn_no, "face_image": face_image_path})
    return {"message": "Face added successfully", "user_id": user_id, "face_image": face_image_path}

@app.post("/recognize_face")
async def recognize_face(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        query_embedding, _ = get_face_embedding(file_path)
    except ValueError as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    os.remove(file_path)
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_embedding.tolist(), limit=1
    )
    
    if not search_results.points:
        return {"message": "No matching face found"}
    
    best_match = search_results.points[0]
    user_data = users_collection.find_one({"_id": best_match.id})
    
    return {"user_id": best_match.id, "name": user_data["name"], "prn_no": user_data["prn_no"]}

@app.get("/capture_photo")
def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture photo")
    file_path = "captured_face.jpg"
    cv2.imwrite(file_path, frame)
    return {"message": "Photo captured", "file_path": file_path}

# Additional routes for real-time recognition and live streaming can be added.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

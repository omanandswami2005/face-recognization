import cv2
import numpy as np
import face_recognition
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ----- Configuration -----
# Qdrant settings:
# If using Qdrant Cloud, replace with your cloud URL and API key.
# For local Docker, use: url="http://localhost:6333" and omit api_key.
QDRANT_URL = "<URL>"  # Your Qdrant Cloud URL
API_KEY = "<API_key>"  # Replace with your actual API key

# Collection name and vector dimensions
COLLECTION_NAME = "face_attendance"
VECTOR_DIM = 128  # face_recognition produces 128-dim embeddings

# ----- Step 1: Connect to Qdrant -----
if API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)

# Create the collection if it does not exist
if not client.collection_exists(COLLECTION_NAME):
    print(f"Collection {COLLECTION_NAME} does not exist. Creating...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),

    )

# ----- Step 2: Define a Function to Get Face Embedding -----
def get_face_embedding(image_path):
    # Load the image using face_recognition (which uses RGB format)
    """
    Loads an image from a file, detects a face in it, and computes the 128-d face embedding.

    Args:
        image_path (str): Path to the image file

    Returns:
        face_embeddings (numpy.ndarray): 128-d face embedding

    Raises:
        ValueError: If no face is detected in the image
    """
    image = face_recognition.load_image_file(image_path)
    # Detect face locations (assuming one face per image)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise ValueError(f"No face detected in {image_path}")
    # Compute the face Embeddings (128-d vector)
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)
    face_embeddings = face_encoding[0]  # Assuming one face per image
    return face_embeddings

# ----- Step 3: Insert a Face Embedding into Qdrant -----
# Example: add a known person’s face (for instance, "Alice")
# try:
#     balaji_embedding = get_face_embedding("omi2.jpg")  # Replace with the path to your image
# except Exception as e:
#     print(e)
#     exit()

# # # Optionally, print the embedding for debugging
# # print("balaji embedding:", alice_embedding)

# # # # Upsert the embedding with a unique ID and payload data
# point = PointStruct(
#     id=1,
#     vector=balaji_embedding,
#     payload={"name": "om", "description": "Registered face for attendance"}
# )
# #insert +update = upsert
# client.upsert(collection_name=COLLECTION_NAME, points=[point])
# print("Inserted om face embedding into Qdrant DB.")

# ----- Step 4: Query by Comparing a New Face -----
# Assume we have a new image (e.g., "balaji1.jpg") for verification
try:
    query_embedding = get_face_embedding("omi2.jpg")  # Replace with the path to your image
except Exception as e:
    print(e)
    exit()

# Search for the most similar face in the collection using the correct search method
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=1  # Looking for the closest match
)

# print("hits:", hits)

if search_results:
    # Assuming search_results is your list of scored points
   for point in search_results.points:
        print("ID:", point.id)
        print("Score:", point.score)
        print("Payload:", point.payload)
        # print("Shard Key:", point.shard_key)
        # print("Order Value:", point.order_value)
        # print("-" * 40)
else:
    print("No matching face found.")


# ----- Optional: Real-Time Webcam Recognition -----
# Uncomment the following block if you want to run real-time face recognition.
# It will capture from your webcam, encode the face, and try to match against the stored data.

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame")
#         break

#     # Resize frame for faster processing
#     small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Balanced speed vs accuracy
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

#     # Detect faces and compute embeddings
#     face_locations = face_recognition.face_locations(rgb_small_frame)
#     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#     for face_encoding, face_location in zip(face_encodings, face_locations):
#         # Search in Qdrant using `query_points`
#         search_results = client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=face_encoding.tolist(),
#             limit=1
#         )

#         name = "Unknown"  # Default label

#         if search_results.points:  # If a match is found
#             best_match = search_results.points[0]
#             if best_match.score > 0.95:  # Adjust threshold for accuracy
#                 name = best_match.payload.get("name", "Unknown")

#         # Scale face location back to original frame size
#         top, right, bottom, left = [x * 2 for x in face_location]

#         # Draw rectangle & name label
#         if name != "Unknown":
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        

#     # Display the frame
#     cv2.imshow("Webcam Face Recognition", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
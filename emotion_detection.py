import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import deque

# Define emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'confused']

# Define transforms with better preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Face alignment function
def align_face(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None, None
    
    # Get the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face ROI
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect eyes for alignment
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_roi)
    
    if len(eyes) >= 2:
        try:
            # Get the two largest eyes
            eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]
            
            # Calculate eye centers
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (float(x + ex + ew//2), float(y + ey + eh//2))
                eye_centers.append(eye_center)
            
            # Calculate angle for alignment
            left_eye, right_eye = eye_centers
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                        right_eye[0] - left_eye[0]))
            
            # Calculate center point
            center_x = (left_eye[0] + right_eye[0]) / 2
            center_y = (left_eye[1] + right_eye[1]) / 2
            center = (center_x, center_y)
            
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
            
            # Extract aligned face
            faces = face_cascade.detectMultiScale(aligned, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                face_roi = aligned[y:y+h, x:x+w]
                return face_roi, (x, y, w, h)
        except Exception as e:
            print(f"Warning: Face alignment failed: {str(e)}")
            # Return unaligned face if alignment fails
            return face_roi, (x, y, w, h)
    
    return face_roi, (x, y, w, h)

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EmotionDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionNet().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load('emotion_model.pth'))
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print("Error: Could not load the model. Please train the model first using train_model.py")
            print(f"Error details: {str(e)}")
            raise e

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize temporal smoothing
        self.emotion_history = deque(maxlen=5)  # Store last 5 predictions
        self.confidence_threshold = 0.6  # Minimum confidence threshold

    def smooth_predictions(self, predictions):
        """Apply temporal smoothing to predictions"""
        self.emotion_history.append(predictions)
        
        if len(self.emotion_history) < 3:
            return predictions
            
        # Count occurrences of each emotion in history
        emotion_counts = {}
        for pred in self.emotion_history:
            for emotion, conf in pred:
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1
                
        # Get most frequent emotion
        most_frequent = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Return predictions with most frequent emotion first
        result = []
        for emotion, conf in predictions:
            if emotion == most_frequent:
                result.insert(0, (emotion, conf))
            else:
                result.append((emotion, conf))
                
        return result

    def detect_emotion(self, image):
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Align face
        face_roi, face_coords = align_face(image, self.face_cascade)
        
        if face_roi is None:
            return "No face detected"
        
        # Convert to PIL Image
        face_pil = Image.fromarray(face_roi)
        
        # Apply transforms
        face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            # Convert to list of (emotion, confidence) tuples
            predictions = []
            for prob, idx in zip(top3_prob[0], top3_indices[0]):
                emotion = EMOTIONS[idx.item()]
                confidence = prob.item()
                predictions.append((emotion, confidence))
            
            # Apply temporal smoothing
            smoothed_predictions = self.smooth_predictions(predictions)
            
            # Filter by confidence threshold
            result = []
            for emotion, conf in smoothed_predictions:
                if conf >= self.confidence_threshold:
                    result.append(f"{emotion} ({conf*100:.1f}%)")
            
            if not result:
                return "Uncertain"
                
            return " | ".join(result)

def detect_emotion():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionNet().to(device)
    
    try:
        model.load_state_dict(torch.load('emotion_model.pth'))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print("Error: Could not load the model. Please train the model first using train_model.py")
        print(f"Error details: {str(e)}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize emotion detector
    detector = EmotionDetector()

    print("Starting emotion detection... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Detect emotion
        result = detector.detect_emotion(frame)
        
        # Draw results
        if result != "No face detected" and result != "Uncertain":
            # Get face coordinates
            face_roi, (x, y, w, h) = align_face(frame, face_cascade)
            if face_roi is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_emotion() 
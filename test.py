from ultralytics import YOLO
import cv2
import json
import os
 
# Loading the pre-trained model
model = YOLO('/home/arpit-srivastava/Desktop/logo_detection/best(1).pt')

# Path to the input video
input_path = '/home/arpit-srivastava/Desktop/logo_detection/videoplayback (2).mp4'

# Path to save the output video
output_video_name = 'output.mp4'
output_video_path = f'/home/arpit-srivastava/Desktop/logo_detection/{output_video_name}'

# Opening the input video  
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initializing to store the results
results_dict = {
    "Pepsi_pts": [],

    "CocaCola_pts": []
}

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame using the model
    results = model(frame)

    # Get the current timestamp in seconds
    timestamp = frame_number / fps

    # Process the results for the current frame
    for result in results:
        for obj in result.boxes:
            label = model.names[int(obj.cls)]
            if label in ["Pepsi", "CocaCola"]:
                # Get object bounding box coordinates
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())

                # Calculate size and distance from the center
                size = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance_from_center = ((center_x - width / 2) * 2 + (center_y - height / 2) * 2) ** 0.5

                # Appending timestamp to the appropriate list
                if label == "Pepsi":
                    results_dict["Pepsi_pts"].append(timestamp)
                elif label == "CocaCola":
                    results_dict["CocaCola_pts"].append(timestamp)

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} Size: {size} Dist: {distance_from_center:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    frame_number += 1

# Release the video capture and writer objects
cap.release()
out.release()

# Save the results dictionary as a JSON file
results_json_path = os.path.join(os.path.dirname(output_video_path), 'results.json')
with open(results_json_path, 'w') as f:
    json.dump(results_dict, f, indent=4)

print(f"Processed video saved at {output_video_path}")
print(f"Results saved at {results_json_path}")

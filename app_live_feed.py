import cv2
import numpy as np

def display_live_feed(stream_url):
    """
    Display a live feed from a streaming URL.
    
    Args:
        stream_url: URL of the RTSP stream
    """
    print(f"Connecting to stream: {stream_url}")
    print("Initializing video capture... (this may take a few seconds)")
    
    cap = cv2.VideoCapture(stream_url)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Check if stream opened successfully
    if not cap.isOpened():
        print("Error: Could not open stream. Check the URL and network connection.")
        print("\nAvailable streams:")
        print("  rtsp://10.149.73.176:8080/h264.sdp")
        print("  rtsp://10.149.73.176:8080/h264_aac.sdp")
        print("  rtsp://10.149.73.176:8080/h264_opus.sdp")
        print("  rtsp://10.149.73.176:8080/h264_ulaw.sdp")
        return
    
    print("Stream connected! Press 'q' to quit.")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame. Stream may have disconnected.")
            break
        
        frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Live Feed', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Closing stream... Total frames received: {frame_count}")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # RTSP stream URLs - choose one:
    # stream_url = "rtsp://10.149.73.176:8080/h264.sdp"
    stream_url = "rtsp://10.149.73.176:8080/h264.sdp"
    
    display_live_feed(stream_url)

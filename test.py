import cv2
import numpy as np

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Desired display size
display_width = 960
display_height = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to 16-bit if needed
    frame16 = frame.astype(np.uint16) if frame.dtype != np.uint16 else frame

    # Convert 10-bit to 8-bit
    frame8 = (frame16 >> 2).astype(np.uint8)

    # Ensure single channel
    if frame8.ndim == 3:
        frame8 = frame8[:, :, 0]

    # Bayer → RGB
    rgb = cv2.cvtColor(frame8, cv2.COLOR_BAYER_RG2RGB)

    # Resize to smaller window
    rgb_small = cv2.resize(rgb, (display_width, display_height))

    # Show
    cv2.imshow("RGB Small", rgb_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
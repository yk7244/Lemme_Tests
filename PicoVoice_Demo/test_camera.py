import cv2

def test_camera():
    for index in range(0, 3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera at index {index}")
            break
        else:
            print(f"Camera at index {index} not available.")

    if not cap.isOpened():
        print("No available camera found.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        cv2.imshow('Test Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()

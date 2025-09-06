import cv2
import os
import time

directory = 'Images(Dataset) 400x300/'
print(os.getcwd())

# Create main directories
if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

cap = cv2.VideoCapture(0)


def capture_images(letter, folder_name):
    count = len(os.listdir(os.path.join(directory, folder_name)))
    print(
        f"[INFO] Starting capture for '{letter}' in 10 seconds. Get ready...")
    time.sleep(10)   # wait before starting

    for i in range(40):  # capture 40 images
        _, frame = cap.read()
        roi = frame[50:400, 0:400]

        filename = os.path.join(directory, folder_name, f"{count+i}.jpg")
        cv2.imwrite(filename, roi)

        cv2.rectangle(frame, (0, 50), (400, 400), (255, 255, 255), 2)
        cv2.imshow("ROI", roi)
        cv2.imshow("data", frame)

        cv2.waitKey(100)  # small delay (100ms) between captures

    print(f"[INFO] Finished capturing 40 images for '{letter}'.")


while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 50), (400, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF in range(ord('a'), ord('z')+1):
        letter = chr(interrupt & 0xFF)
        folder_name = letter.upper()
        capture_images(letter, folder_name)

    if interrupt & 0xFF == ord('.'):
        capture_images('blank', 'blank')

    if interrupt & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

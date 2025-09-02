import cv2
import os

directory = 'Images(Dataset) 400x300/'
print(os.getcwd())

if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    count = {
        'a': len(os.listdir(os.path.join(directory, "A"))),
        'b': len(os.listdir(os.path.join(directory, "B"))),
        'c': len(os.listdir(os.path.join(directory, "C"))),
        'd': len(os.listdir(os.path.join(directory, "D"))),
        'e': len(os.listdir(os.path.join(directory, "E"))),
        'f': len(os.listdir(os.path.join(directory, "F"))),
        'g': len(os.listdir(os.path.join(directory, "G"))),
        'h': len(os.listdir(os.path.join(directory, "H"))),
        'i': len(os.listdir(os.path.join(directory, "I"))),
        'j': len(os.listdir(os.path.join(directory, "J"))),
        'k': len(os.listdir(os.path.join(directory, "K"))),
        'l': len(os.listdir(os.path.join(directory, "L"))),
        'm': len(os.listdir(os.path.join(directory, "M"))),
        'n': len(os.listdir(os.path.join(directory, "N"))),
        'o': len(os.listdir(os.path.join(directory, "O"))),
        'p': len(os.listdir(os.path.join(directory, "P"))),
        'q': len(os.listdir(os.path.join(directory, "Q"))),
        'r': len(os.listdir(os.path.join(directory, "R"))),
        's': len(os.listdir(os.path.join(directory, "S"))),
        't': len(os.listdir(os.path.join(directory, "T"))),
        'u': len(os.listdir(os.path.join(directory, "U"))),
        'v': len(os.listdir(os.path.join(directory, "V"))),
        'w': len(os.listdir(os.path.join(directory, "W"))),
        'x': len(os.listdir(os.path.join(directory, "X"))),
        'y': len(os.listdir(os.path.join(directory, "Y"))),
        'z': len(os.listdir(os.path.join(directory, "Z"))),
        'blank': len(os.listdir(os.path.join(directory, "blank")))
    }

    cv2.rectangle(frame, (0, 50), (400, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    # ROI in original color (no grayscale, no resize)
    roi = frame[50:400, 0:400]
    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(os.path.join(directory, 'A', f"{count['a']}.jpg"), roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(os.path.join(directory, 'B', f"{count['b']}.jpg"), roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(directory, 'C', f"{count['c']}.jpg"), roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(os.path.join(directory, 'D', f"{count['d']}.jpg"), roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(os.path.join(directory, 'E', f"{count['e']}.jpg"), roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(os.path.join(directory, 'F', f"{count['f']}.jpg"), roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(os.path.join(directory, 'G', f"{count['g']}.jpg"), roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(os.path.join(directory, 'H', f"{count['h']}.jpg"), roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(os.path.join(directory, 'I', f"{count['i']}.jpg"), roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(os.path.join(directory, 'J', f"{count['j']}.jpg"), roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(os.path.join(directory, 'K', f"{count['k']}.jpg"), roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(os.path.join(directory, 'L', f"{count['l']}.jpg"), roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(os.path.join(directory, 'M', f"{count['m']}.jpg"), roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(os.path.join(directory, 'N', f"{count['n']}.jpg"), roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(os.path.join(directory, 'O', f"{count['o']}.jpg"), roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(os.path.join(directory, 'P', f"{count['p']}.jpg"), roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(os.path.join(directory, 'Q', f"{count['q']}.jpg"), roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(os.path.join(directory, 'R', f"{count['r']}.jpg"), roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(os.path.join(directory, 'S', f"{count['s']}.jpg"), roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(os.path.join(directory, 'T', f"{count['t']}.jpg"), roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(os.path.join(directory, 'U', f"{count['u']}.jpg"), roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(os.path.join(directory, 'V', f"{count['v']}.jpg"), roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(os.path.join(directory, 'W', f"{count['w']}.jpg"), roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(os.path.join(directory, 'X', f"{count['x']}.jpg"), roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(os.path.join(directory, 'Y', f"{count['y']}.jpg"), roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(os.path.join(directory, 'Z', f"{count['z']}.jpg"), roi)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory, 'blank',
                    f"{count['blank']}.jpg"), roi)

cap.release()
cv2.destroyAllWindows()

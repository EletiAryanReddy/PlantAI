import cv2

def open_camera():

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            break

        cv2.imshow(
            "Plant Disease Camera Scanner",
            frame
        )

        key = cv2.waitKey(1)

        # press q to quit
        if key == ord('q'):
            break

    camera.release()

    cv2.destroyAllWindows()
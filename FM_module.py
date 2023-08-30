import time
import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=3, land=False, minDetectionCon=0.5, minTrackerCon=0.5):
        self.imgRGB = None
        self.results = None
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.land = land
        self.minDetectionCon = minDetectionCon
        self.minTrackerCon = minTrackerCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.land, self.minDetectionCon,
                                                 self.minTrackerCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawSpec,
                                               self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # for id numbers in the mesh
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=3, minDetectionCon=0.5, minTrackerCon=0.5)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)  # false to remove mask
        if len(faces) != 0:
            print(faces)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the program
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ==== MODEL SETUP ====
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# ==== TKINTER GUI ====
window = tk.Tk()
window.title("Deteksi Wajah, Usia, dan Gender")
window.configure(bg="#f0f0f0")
window.geometry("700x630")

title_label = tk.Label(window, text="🎯 Deteksi Wajah, Usia & Gender", font=("Helvetica", 16, "bold"), fg="#333", bg="#f0f0f0")
title_label.pack(pady=10)

video_label = tk.Label(window, bg="#222")
video_label.pack(padx=10, pady=10)

is_running = False
cap = None

def detect_faces_from_image(image_path):
    img = cv2.imread(image_path)
    resultImg = img.copy()

    blob = cv2.dnn.blobFromImage(resultImg, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * img.shape[1])
            y1 = int(detections[0, 0, i, 4] * img.shape[0])
            x2 = int(detections[0, 0, i, 5] * img.shape[1])
            y2 = int(detections[0, 0, i, 6] * img.shape[0])
            face = img[y1:y2, x1:x2]

            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob_face)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob_face)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(resultImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resultImg, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Tampilkan di GUI
    rgb_img = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb_img)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        detect_faces_from_image(file_path)

def detect_and_display():
    global cap, is_running
    if not is_running or cap is None:
        return
    ret, frame = cap.read()
    if not ret:
        return

    resultImg = frame.copy()
    blob = cv2.dnn.blobFromImage(resultImg, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            face = frame[y1:y2, x1:x2]

            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob_face)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob_face)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(resultImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resultImg, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    img = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, detect_and_display)

def start_camera():
    global is_running, cap
    if not is_running:
        cap = cv2.VideoCapture(0)
        is_running = True
        detect_and_display()

def stop_camera():
    global is_running, cap
    is_running = False
    if cap is not None:
        cap.release()
        cap = None
        video_label.config(image='')

# ==== TOMBOL ====
btn_frame = tk.Frame(window, bg="#f0f0f0")
btn_frame.pack(pady=15)

btn_start = tk.Button(btn_frame, text="▶ Mulai Kamera", command=start_camera, width=18, bg='#28a745', fg='white', font=("Helvetica", 10, "bold"))
btn_start.grid(row=0, column=0, padx=5)

btn_stop = tk.Button(btn_frame, text="■ Berhenti", command=stop_camera, width=18, bg='#dc3545', fg='white', font=("Helvetica", 10, "bold"))
btn_stop.grid(row=0, column=1, padx=5)

btn_image = tk.Button(btn_frame, text="📷 Deteksi dari Gambar", command=browse_image, width=25, bg='#007bff', fg='white', font=("Helvetica", 10, "bold"))
btn_image.grid(row=1, column=0, columnspan=2, pady=10)

window.mainloop()

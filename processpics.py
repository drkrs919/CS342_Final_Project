import cv2

# Define the desired dimensions for the cropped and resized images
crop_size = (500, 500)

def crop_profs(gray = False):
    for i in range(1, 99):
        image_path = f'./CS_prof_images/image_{i}.jpg'
        img = cv2.imread(image_path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_image = img[y:y+h, x:x+w]
            resized_img = cv2.resize(face_image, crop_size)
            rgb_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            output_path = f'./profdatacolor/Faces/image_{i}.jpg'
            if gray:
                output_path = f'./profdatagray/Faces/image_{i}.jpg'
            Image.fromarray(rgb_resized_img).save(output_path)
        else:
            print(f"No face detected in image {i}. Skipping.")

def crop_database(gray = False):
    for i in range(1, 13234):
        image_path = f'./lfw/image_{i}.jpg'
        img = cv2.imread(image_path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_image = img[y:y+h, x:x+w]
            resized_img = cv2.resize(face_image, crop_size)
            rgb_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            output_path = f'./lfwdatacolor/Faces/image_{i}.jpg'
            if gray:
                output_path = f'./lfwdatagray/Faces/image_{i}.jpg'
            Image.fromarray(rgb_resized_img).save(output_path)
        else:
            print(f"No face detected in image {i}. Skipping.")


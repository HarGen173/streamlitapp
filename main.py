import streamlit as st
from PIL import Image  # Import the PIL library

import cv2

import numpy as np

def process(nimg, color, kernel_size=1):
    img = cv2.cvtColor(nimg, cv2.COLOR_RGB2GRAY)

    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    if color == "red":
        lower = np.array([155,25,0])
        upper = np.array([179,255,255])
    else:
        lower = (36, 25, 25)
        upper = (96, 255,255)

    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -1.5)

    hsv = cv2.cvtColor(nimg, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    t, mask = cv2.threshold(mask,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    imask = mask>0
    colored = np.zeros_like(img, np.uint8)
    colored[imask] = th[imask]

    th = colored

    kernel_1 = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_2 = np.ones((kernel_size, kernel_size), np.uint8)

    th = cv2.erode(th, kernel_1, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th)

    blobs = []

    cimg = nimg.copy()

    for i, blob in enumerate(centroids):
        if stats[i, cv2.CC_STAT_AREA] < 15: continue
        blobs.append(blob)
        x, y = blob
        cv2.circle(cimg,(int(x),int(y)),2,(0,0,255),3)

    return cimg

def main():

    st.title("Тальбограммометр")

    # Sidebar for file uploader
    with st.sidebar:
        uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
        option = st.selectbox("Выберите цвет решетки",("red", "green"))
        kernel = st.slider("Выберите размер фильтра", 1, 11, 3, 2)
        button = st.button("Найти точки")

    # Main area
    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.header("Исходное изображение")
        if uploaded_file is not None:
            # Display the uploaded image in the first column
            try:
                image = Image.open(uploaded_file)  # Use PIL to open the image
                st.image(image, use_container_width=True)
                st.session_state.original_image = image # Store original image in session state

            except Exception as e:
                st.error(f"Ошибка: Невозможно открыть изображение.  {e}")
        else:
            st.write("Загрузите изображение.")

    with col2:
        st.header("Изображение с точками")
        if button:  # Button to trigger processing
            if 'original_image' in st.session_state:
                # Process the image (in this case, just show it again)
                pil_image = Image.open(uploaded_file).convert('RGB')
                open_cv_image = np.array(pil_image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                processed_image = process(open_cv_image, option, kernel)
                st.image(processed_image, use_container_width=True)
            else:
                st.write("Пожалуйста, сначала выберите изображение.")


if __name__ == "__main__":
    main()
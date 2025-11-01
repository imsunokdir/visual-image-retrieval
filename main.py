import streamlit as st
import cv2
import numpy as np
import random
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="Visual Image Retrieval", layout="wide")
st.title("🔍 Visual Image Retrieval using ORB Features")

# --- Load custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./style.css")

# Upload images
target_file = st.file_uploader("📸 Upload Target Image", type=["jpg", "png", "jpeg"])
query_files = st.file_uploader("🖼️ Upload Query Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if target_file and query_files:
    # Load and preprocess target image
    target = np.array(Image.open(target_file).convert("RGB"))
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    # ORB and matcher
    orb = cv2.ORB_create(nfeatures=15000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Work on a copy of target
    result = target.copy()

    for query_file in query_files:
        query_name = query_file.name
        query = np.array(Image.open(query_file).convert("RGB"))
        query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)

        # Detect features
        kp1, des1 = orb.detectAndCompute(query_gray, None)
        kp2, des2 = orb.detectAndCompute(target_gray, None)

        if des1 is None or des2 is None:
            st.warning(f"⚠️ No features detected in {query_name}")
            continue

        # Match features
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if matrix is not None:
                h, w = query_gray.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                color = [random.randint(50, 255) for _ in range(3)]
                cv2.polylines(result, [np.int32(dst)], True, color, 3, cv2.LINE_AA)

                # Confidence calculation
                inliers = np.sum(mask)
                confidence = min(100, (inliers / len(good)) * 100)
                conf_text = f"{query_name}: {confidence:.1f}%"

                x, y = np.int32(dst[0][0])
                cv2.putText(result, conf_text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            else:
                st.warning(f"⚠️ {query_name}: Homography failed.")
        else:
            st.warning(f"⚠️ {query_name}: Not enough matches ({len(good)})")

    st.image(result, caption="Detected Objects with Confidence", use_container_width=True)
else:
    st.info("👆 Please upload a target image and at least one query image to begin.")

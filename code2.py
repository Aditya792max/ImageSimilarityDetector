import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from PIL import Image
import clip
import matplotlib.pyplot as plt




# ---------- LOAD IMAGES ----------
original = cv2.imread('/Users/adityavikramkirtania/Desktop/PythonDataAnalytics/ImageGames/imageSimilarityComparator/Signature1.jpeg')
duplicate = cv2.imread('/Users/adityavikramkirtania/Desktop/PythonDataAnalytics/ImageGames/imageSimilarityComparator/Signature2.jpeg')

# Resize to same size
duplicate = cv2.resize(duplicate, (original.shape[1], original.shape[0]))

print("✅ Images loaded successfully.\n")

# ---------- 1️⃣ SSIM (Structural Similarity) ----------
grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)

ssim_score, diff = ssim(grayA, grayB, full=True)
print(f"🧩 SSIM Score: {ssim_score:.4f}")
if ssim_score > 0.9:
    print("➡️ Images are VERY similar (SSIM)")
elif ssim_score > 0.5:
    print("➡️ Images are SOMEWHAT similar (SSIM)")
else:
    print("➡️ Images are DIFFERENT (SSIM)")
print("\n------------------------------------\n")

# ---------- 2️⃣ HISTOGRAM COMPARISON ----------
hsvA = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
hsvB = cv2.cvtColor(duplicate, cv2.COLOR_BGR2HSV)

histA = cv2.calcHist([hsvA], [0, 1], None, [50, 60], [0, 180, 0, 256])
histB = cv2.calcHist([hsvB], [0, 1], None, [50, 60], [0, 180, 0, 256])

cv2.normalize(histA, histA)
cv2.normalize(histB, histB)

hist_similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
print(f"🎨 Histogram Similarity: {hist_similarity:.4f}")
if hist_similarity > 0.9:
    print("➡️ Images are VERY similar (Histogram)")
elif hist_similarity > 0.6:
    print("➡️ Images are SOMEWHAT similar (Histogram)")
else:
    print("➡️ Images are DIFFERENT (Histogram)")
print("\n------------------------------------\n")

# ---------- 3️⃣ ORB FEATURE MATCHING ----------
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = orb.detectAndCompute(cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY), None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
match_ratio = len(matches) / min(len(kp1), len(kp2))
print(f"🔍 ORB Feature Match Ratio: {match_ratio:.4f}")
if match_ratio > 0.3:
    print("➡️ Images are SIMILAR (ORB Features)")
else:
    print("➡️ Images are DIFFERENT (ORB Features)")
print("\n------------------------------------\n")

# ---------- 4️⃣ DEEP LEARNING (CLIP COSINE SIMILARITY) ----------
print("🧠 Loading CLIP model (this might take a few seconds)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image1 = preprocess(Image.open('/Users/adityavikramkirtania/Desktop/PythonDataAnalytics/ImageGames/imageSimilarityComparator/Image1.jpeg')).unsqueeze(0).to(device)
image2 = preprocess(Image.open('/Users/adityavikramkirtania/Desktop/PythonDataAnalytics/ImageGames/imageSimilarityComparator/Image2.jpeg')).unsqueeze(0).to(device)

with torch.no_grad():
    features1 = model.encode_image(image1)
    features2 = model.encode_image(image2)

similarity = torch.cosine_similarity(features1, features2).item()
print(f"🤖 CLIP Cosine Similarity: {similarity:.4f}")
if similarity > 0.9:
    print("➡️ Images are VERY similar (CLIP)")
elif similarity > 0.7:
    print("➡️ Images are SOMEWHAT similar (CLIP)")
else:
    print("➡️ Images are DIFFERENT (CLIP)")

print("\n✅ Comparison complete!")

# ---------- 5️⃣ VISUAL COMPARISON GRAPHS ----------
# Collect all scores
methods = ['SSIM', 'Histogram', 'ORB', 'CLIP']
scores = [ssim_score, hist_similarity, match_ratio, similarity]

# Plot as bar chart
plt.figure(figsize=(10,6))
bars = plt.bar(methods, scores, color=['skyblue','orange','lightgreen','violet'])
plt.ylim(0,1.1)
plt.title("Image Similarity Comparison Across Methods", fontsize=15)
plt.ylabel("Similarity Score (0 to 1)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", 
             ha='center', fontsize=11, fontweight='bold')

plt.show()







# Normalize so total adds up to 100%
percentages = [score / sum(scores) * 100 for score in scores]

plt.figure(figsize=(8,8))
colors = ['skyblue', 'orange', 'lightgreen', 'violet']
explode = [0.05 if s == max(scores) else 0 for s in scores]

plt.pie(percentages, labels=methods, autopct='%1.1f%%', startangle=140, 
        colors=colors, shadow=True, explode=explode, textprops={'fontsize': 12})
plt.title("Image Similarity Distribution Across Methods", fontsize=15, fontweight='bold')
plt.show()














# ---------- 6️⃣ SHOW INPUT IMAGES ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(duplicate, cv2.COLOR_BGR2RGB))
plt.title("Duplicate Image")
plt.axis("off")
plt.show()

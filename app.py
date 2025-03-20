import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import openai
import os
import base64
from PIL import Image

# ✅ Model Path
MODEL_PATH = "88_categories.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ OpenAI API Key not found! Set it as an environment variable.")

# ✅ OpenAI API Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Define the 88 Car Part Categories (Strict List)
CAR_PARTS = [
    "AC COMPRESSOR", "ACCELERATOR PEDAL ASSEMBLY", "AIR FILTER", "ALTERNATOR",
    "BACKUP CAMERA", "BALL JOINT", "BATTERY", "BLOWER MOTOR", "BONNET",
    "BRAKE BOOSTER", "BRAKE CALIPER", "BRAKE DISC", "BRAKE LINING", "BRAKE PAD",
    "BRAKE SHOE", "BULB", "BULB SOCKET", "CABIN FILTER", "CAMSHAFT",
    "CLOCK SPRING", "CLUTCH DISC", "CLUTCH KIT", "CLUTCH RELEASE BEARING",
    "CLUTCH SET", "COIL SPRING", "CONDENSER", "CONNECTING ROD",
    "COVER ASSEMBLY", "CRANKSHAFT", "CYLINDER HEAD", "DICKY SHOCK ABSORBER",
    "DRIVE SHAFT - LH", "ENGINE FLUSH", "EVAPORATOR", "FOG LAMP",
    "FOG LAMP COVER", "FRONT SHOCK ABSORBER", "FRONT STABILIZER LINK",
    "FRONT WHEEL HUB", "FUEL FILTER", "FUEL PUMP", "FUEL TANK CAP", "FUSE",
    "GLOW PLUG", "HALF ENGINE", "HEAD LAMP BULB", "HEAD LIGHT", "HORN",
    "IGNITION CABLE", "IGNITION COIL", "INJECTOR", "INSTRUMENT CLUSTER",
    "JACK", "LEAF SPRING", "LOWER TRACK CONTROL ARM", "OIL FILTER", "OIL SUMP",
    "OXYGEN SENSOR", "PISTON", "POWER WINDOW SWITCH", "RADIATOR", "RADIATOR CAP",
    "RADIATOR FAN ASSEMBLY", "RADIATOR HOSE", "REAR SHOCK ABSORBER",
    "REAR WHEEL HUB ASSEMBLY", "SENSOR", "SILENCER", "SPARK PLUG", "STARTER",
    "STRUT MOUNTING", "SVM - LH", "SVM - RH", "TAIL LAMP ASSEMBLY - LH",
    "TAIL LAMP ASSEMBLY - RH", "TENSIONER", "THERMOSTAT ASSEMBLY",
    "THROTTLE BODY CLEANER", "TORQUE CONVERTER", "WATER PUMP", "WHEEL ALLOY",
    "WHEEL CAP", "WHEEL COVER", "WIPER BLADE", "WIPER MOTOR ASSEMBLY",
    "WIPER TANK", "WIRING HARNESS KIT", "WIRING KIT WO RELAY"
]

# ✅ File & Folder Paths
CSV_FILE = "result.csv"
HTML_FILE = "prediction.html"
SAVED_IMAGES_DIR = "saved_images"

# ✅ Ensure required files/folders exist
os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Image Path", "Model Prediction", "ChatGPT Prediction", "Confidence Score", "Match Status"]).to_csv(CSV_FILE, index=False)

# ✅ Load previous predictions
df_results = pd.read_csv(CSV_FILE)
existing_images = set(df_results["Image Path"])
results = df_results.to_dict("records")

# ✅ Image Preprocessing for Model
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Convert Image to Base64 for ChatGPT
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ✅ ChatGPT Prediction (STRICT to 88 Categories)
def chatgpt_predict(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"Identify the car part in the image. Only return the name of the car part from this strict list: {', '.join(CAR_PARTS)}. Do not return any explanation, only the name."},
            {"role": "user", "content": [
                {"type": "text", "text": "Classify this car part strictly from the given categories."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )

    predicted_part = response.choices[0].message.content.strip().upper()

    return predicted_part if predicted_part in CAR_PARTS else "UNKNOWN"

# ✅ Streamlit UI
st.title("🚗 Car Parts Classification & Analysis")
st.write("Upload images to classify car parts using AI.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    new_results = []
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_path = os.path.join(SAVED_IMAGES_DIR, uploaded_file.name)

        if image_path in existing_images:
            continue  

        image.save(image_path)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        model_predicted_class = CAR_PARTS[np.argmax(prediction)]
        confidence_score = np.max(prediction) * 100
        chatgpt_predicted_class = chatgpt_predict(image_path)

        match_status = "✅ Match" if model_predicted_class == chatgpt_predicted_class else "❌ Mismatch"

        result_entry = {"Image Path": image_path, "Model Prediction": model_predicted_class, "ChatGPT Prediction": chatgpt_predicted_class, "Confidence Score": round(confidence_score, 2), "Match Status": match_status}
        new_results.append(result_entry)
        results.append(result_entry)

    pd.DataFrame(new_results).to_csv(CSV_FILE, mode='a', header=False, index=False)

# ✅ Generate Updated HTML Report (Grid Layout: 5 Predictions per Row)
with open(HTML_FILE, "w") as f:
    f.write("""
    <html>
    <head>
        <title>Predictions</title>
        <style>
            .container { display: flex; flex-wrap: wrap; justify-content: center; }
            .card { width: 18%; border: 1px solid black; padding: 10px; margin: 10px; text-align: center; 
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1); background-color: #f9f9f9; }
            img { max-width: 100%; height: auto; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        </style>
    </head>
    <body><h1>🚗 Car Parts Classification Results</h1><div class="container">
    """)

    for result in results:
        f.write(f"""
            <div class="card">
                <img src="{result['Image Path']}" alt="Car Part">
                <h2>{result["Model Prediction"]}</h2>
                <p><b>ChatGPT:</b> {result["ChatGPT Prediction"]}</p>
                <p><b>Confidence:</b> {result["Confidence Score"]}%</p>
                <p><b>Status:</b> {result["Match Status"]}</p>
            </div>
        """)

    f.write("</div></body></html>")

# ✅ Download Buttons
st.download_button("📥 Download HTML Report", open(HTML_FILE, "rb"), "prediction.html", "text/html")
st.download_button("📥 Download Results CSV", open(CSV_FILE, "rb"), "result.csv", "text/csv")


# ✅ Match vs. Mismatch Overview
total_predictions = len(df_results)
total_match = len(df_results[df_results["Match Status"] == "✅ Match"])
total_mismatch = len(df_results[df_results["Match Status"] == "❌ Mismatch"])
match_percentage = (total_match / total_predictions) * 100 if total_predictions > 0 else 0
mismatch_percentage = (total_mismatch / total_predictions) * 100 if total_predictions > 0 else 0

st.write(f"🔹 **Total Predictions:** {total_predictions}")
st.write(f"✅ **Matches:** {total_match} ({match_percentage:.2f}%)")
st.write(f"❌ **Mismatches:** {total_mismatch} ({mismatch_percentage:.2f}%)")

# ✅ Bar Chart: Performance per Category
st.subheader("📈 Category-Wise Performance")

category_performance = df_results.groupby("Model Prediction")["Match Status"].value_counts().unstack().fillna(0)
category_performance["Total Images"] = category_performance.sum(axis=1)
category_performance["Match %"] = (category_performance["✅ Match"] / category_performance["Total Images"]) * 100

# ✅ Display the table
st.dataframe(category_performance.sort_values("Total Images", ascending=False))

# ✅ Pie Chart: Match vs. Mismatch
st.subheader("🎯 Match vs. Mismatch Distribution")

plt.figure(figsize=(5, 5))
plt.pie([total_match, total_mismatch], labels=["Match", "Mismatch"], autopct="%1.1f%%", colors=["green", "red"])
st.pyplot(plt)

# ✅ List of Weak Performing Categories (Mismatch > 30%)
weak_categories = category_performance[category_performance["Match %"] < 70].index.tolist()

if weak_categories:
    st.subheader("⚠️ Categories That Need Improvement")
    st.write("🔍 These categories had **more than 30% mismatches**:")
    for category in weak_categories:
        st.write(f"- {category}")

# ✅ List of Strong Performing Categories (Match > 90%)
strong_categories = category_performance[category_performance["Match %"] > 90].index.tolist()

if strong_categories:
    st.subheader("🏆 Best Performing Categories")
    st.write("✔ These categories had **more than 90% match rate**:")
    for category in strong_categories:
        st.write(f"- {category}")

# ✅ Final Summary
st.subheader("📌 Summary")
st.write("""
- **High Match Categories**: Model is performing well on these.
- **Low Match Categories**: These require more training data.
- **Use this report to refine dataset and improve weaker categories.**
""")

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="FoodFresh AI",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CNN_PATH = os.path.join(MODEL_DIR, "food_freshness_cnn.keras")
MOBILE_PATH = os.path.join(MODEL_DIR, "food_freshness_mobilenet.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""

<style>
    /* APP BACKGROUND AND DEFAULT TEXT */
    .stApp {
        background-color: #fffafa;
        color: #333333 !important;
        --text-color: #333333;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #8b1725;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* PAGE HEADINGS */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #941d2c !important;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777777 !important;
        margin-bottom: 25px;
    }

    /* ALL MAIN PAGE TEXT */
    section.main,
    section.main .block-container {
        color: #333333 !important;
    }

    section.main h1,
    section.main h2,
    section.main h3,
    section.main h4 {
        color: #941d2c !important;
    }

    section.main p,
    section.main li,
    section.main label,
    section.main [data-testid="stMarkdownContainer"],
    section.main [data-testid="stCaptionContainer"],
    section.main [data-testid="stText"],
    section.main [data-testid="stMetricLabel"],
    section.main [data-testid="stMetricValue"] {
        color: #444444 !important;
    }

    /* CUSTOM CARDS */
    .custom-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #f0dadd;
        box-shadow: 0 3px 12px rgba(100, 0, 0, 0.05);
        margin-bottom: 15px;
        color: #444444 !important;
    }

    .custom-card p,
    .custom-card span,
    .custom-card div {
        color: #555555 !important;
        font-size: 16px;
        line-height: 1.7;
    }

    .custom-card h2,
    .custom-card h3,
    .card-heading,
    .section-heading {
        color: #941d2c !important;
    }

    .card-heading {
        font-size: 19px;
        font-weight: 700;
    }

    .big-number {
        color: #941d2c !important;
        font-size: 32px;
        font-weight: 800;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 750;
        margin-top: 15px;
    }

    .small-note {
        color: #777777 !important;
        font-size: 13px;
    }

    /* TABS AND EXPANDERS */
    section.main button[data-baseweb="tab"] {
        color: #444444 !important;
    }

    section.main [data-testid="stExpander"] summary,
    section.main [data-testid="stExpander"] summary p {
        color: #444444 !important;
    }

    /* INPUTS */
    section.main input,
    section.main textarea {
        color: #333333 !important;
    }

    /* BUTTONS */
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background-color: #941d2c;
        border: none;
        color: white !important;
        border-radius: 9px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #6f101d;
        color: white !important;
    }

/* PREDICTION PAGE - LABEL COLOR FIX */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] span,
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stFileUploader"] label {
    color: #941d2c !important;
    -webkit-text-fill-color: #941d2c !important;
    opacity: 1 !important;
}

/* LABEL TEXT VISIBILITY */
.stApp [data-testid="stWidgetLabel"] * {
    color: #941d2c !important;
    -webkit-text-fill-color: #941d2c !important;
}

</style>

""", unsafe_allow_html=True)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def section_title(title, subtitle=None):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def metric_card(title, value, description):
    st.markdown(
        f"""
        <div class="custom-card">
            <div class="card-heading">{title}</div>
            <div class="big-number">{value}</div>
            <div class="small-note">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 4. MODEL FILE CHECK
# =====================================================

required_files = [
    CNN_PATH,
    MOBILE_PATH,
    CLASS_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error("Required project files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# =====================================================
# 5. LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    cnn_model = tf.keras.models.load_model(
        CNN_PATH,
        compile=False
    )

    mobile_model = tf.keras.models.load_model(
        MOBILE_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        safe_mode=False,
        compile=False
    )

    with open(CLASS_PATH, "r") as f:
        classes = json.load(f)

    return cnn_model, mobile_model, classes


try:
    cnn_model, mobile_model, class_names = load_models()
    models_loaded = True

except Exception as error:
    models_loaded = False
    st.error("Model loading failed.")
    st.exception(error)


# =====================================================
# 6. FEEDBACK STORAGE
# =====================================================

def save_feedback(name, category, rating, message):

    new_feedback = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name": name if name else "Anonymous",
        "Category": category,
        "Rating": rating,
        "Message": message
    }])

    if os.path.exists(FEEDBACK_PATH):

        old_feedback = pd.read_csv(FEEDBACK_PATH)

        updated_feedback = pd.concat(
            [old_feedback, new_feedback],
            ignore_index=True
        )

    else:
        updated_feedback = new_feedback

    updated_feedback.to_csv(
        FEEDBACK_PATH,
        index=False
    )


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🥑 FoodFresh AI")
st.sidebar.caption("Smart Food Freshness Detection")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔍 Predict Freshness",
        "📊 Model Performance",
        "🗂️ Dataset Explorer",
        "💡 Food Tips",
        "📝 Suggestion Box",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Deep Learning")
st.sidebar.caption("Custom CNN | MobileNetV2")


# =====================================================
# 8. DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    section_title(
        "FoodFresh AI 🥑",
        "An intelligent system for fruit and vegetable freshness classification."
    )

    st.markdown("""
    <div class="custom-card">
        <h2>Welcome to FoodFresh AI!</h2>
        <p>
            Detect freshness categories of fruits and vegetables
            using deep learning and computer vision.
            Upload an image, select a trained model and get
            an AI-generated prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Dataset Classes", "20", "Fresh and rotten categories")

    with c2:
        metric_card("Deep Learning Models", "2", "Custom CNN and MobileNetV2")

    with c3:
        metric_card("Food Categories", "2", "Fruits and vegetables")

    with c4:
        metric_card("Input Size", "128×128", "RGB image input")

    st.markdown(
        '<div class="section-heading">Explore the Platform</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#941d2c;">🔍 Freshness Prediction</h3>
            Upload an image and predict its freshness category.
        </div>
        <div class="custom-card">
            <h3 style="color:#941d2c;">📊 Model Performance</h3>
            Explore available model evaluation results.
        </div>
        <div class="custom-card">
            <h3 style="color:#941d2c;">🗂️ Dataset Explorer</h3>
            Explore all 20 dataset categories.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#941d2c;">💡 Food Tips</h3>
            Learn useful food storage and freshness tips.
        </div>
        <div class="custom-card">
            <h3 style="color:#941d2c;">📝 Suggestion Box</h3>
            Share your feedback and feature requests.
        </div>
        <div class="custom-card">
            <h3 style="color:#941d2c;">ℹ️ About Project</h3>
            Understand the project methodology and technology.
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Note: Predictions are image classifications, not food safety tests."
    )


# =====================================================
# 9. PREDICTION PAGE
# =====================================================

elif page == "🔍 Predict Freshness":

    section_title(
        "Freshness Prediction 🔍",
        "Upload an image and let the AI model analyze it."
    )

    if not models_loaded:
        st.error("Models are unavailable. Check the loading error.")

    else:

        model_choice = st.selectbox(
            "Choose your Deep Learning Model",
            ["Custom CNN", "MobileNetV2"]
        )

        uploaded_file = st.file_uploader(
            "Upload a fruit or vegetable image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            try:
                image = Image.open(uploaded_file).convert("RGB")

                left, right = st.columns(2)

                with left:
                    st.image(
                        image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )

                with right:
                    st.markdown("### Image Details")
                    st.write("Width:", image.width, "pixels")
                    st.write("Height:", image.height, "pixels")
                    st.write("Model:", model_choice)
                    st.write("Format:", image.format or "RGB")

                if st.button(
                    "✨ Predict Freshness",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing your image..."):

                        resized = image.resize((128, 128))

                        img_array = np.array(
                            resized,
                            dtype=np.float32
                        )

                        img_array = np.expand_dims(
                            img_array,
                            axis=0
                        )

                        if model_choice == "Custom CNN":

                            model = cnn_model
                            img_array = img_array / 255.0

                        else:
                            model = mobile_model

                            # MobileNetV2 preprocessing is embedded
                            # inside the saved model. Do not repeat it.

                        predictions = model.predict(
                            img_array,
                            verbose=0
                        )[0]

                        if len(predictions) != len(class_names):
                            st.error(
                                "Model output count does not match class names."
                            )
                            st.stop()

                        predicted_index = int(np.argmax(predictions))
                        predicted_class = class_names[predicted_index]

                        confidence = float(
                            predictions[predicted_index]
                        ) * 100

                    st.markdown("---")
                    st.markdown("## Prediction Result")

                    result1, result2 = st.columns(2)

                    with result1:
                        st.markdown("### Predicted Category")
                        st.success(predicted_class)

                    with result2:
                        st.markdown("### Confidence")
                        st.metric(
                            "Model Confidence",
                            f"{confidence:.2f}%"
                        )

                    st.progress(
                        min(max(confidence / 100, 0), 1)
                    )

                    if predicted_class.lower().startswith("fresh"):
                        st.success(
                            "The model classified this image as Fresh."
                        )

                    elif predicted_class.lower().startswith("rotten"):
                        st.warning(
                            "The model classified this image as Rotten."
                        )

                    st.markdown("### Top 5 Predictions")

                    top_indices = np.argsort(predictions)[::-1][:5]

                    top_df = pd.DataFrame({
                        "Class": [
                            class_names[i] for i in top_indices
                        ],
                        "Confidence (%)": [
                            float(predictions[i] * 100)
                            for i in top_indices
                        ]
                    })

                    st.bar_chart(
                        top_df.set_index("Class")
                    )

                    st.dataframe(
                        top_df.style.format({
                            "Confidence (%)": "{:.2f}%"
                        }),
                        use_container_width=True
                    )

                    st.caption(
                        "Predictions depend on image quality and training data. "
                        "This tool cannot determine whether food is safe to eat."
                    )

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)


# =====================================================
# 10. MODEL PERFORMANCE
# =====================================================

elif page == "📊 Model Performance":

    section_title(
        "Model Performance 📊",
        "Compare the trained CNN and MobileNetV2 models."
    )

    st.info(
        "For accurate evaluation, copy your actual model comparison "
        "CSV and classification report into the app's results folder."
    )

    st.markdown("### Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#941d2c;">Custom CNN</h3>
            <p>Convolutional Neural Network trained on the food image dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#941d2c;">MobileNetV2</h3>
            <p>Transfer learning architecture used for image classification.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Available Evaluation Files")

    if os.path.isdir(RESULTS_DIR):

        csv_files = [
            file for file in os.listdir(RESULTS_DIR)
            if file.lower().endswith(".csv")
        ]

        image_files = [
            file for file in os.listdir(RESULTS_DIR)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if csv_files:

            selected_csv = st.selectbox(
                "Select an evaluation CSV file",
                csv_files
            )

            if st.button("Load Evaluation CSV"):

                try:
                    df = pd.read_csv(
                        os.path.join(RESULTS_DIR, selected_csv)
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                except Exception as error:
                    st.error(f"Could not read CSV: {error}")

        else:
            st.warning(
                "No evaluation CSV found in the results folder."
            )

        if image_files:

            st.markdown("### Confusion Matrices and Evaluation Charts")

            for image_file in image_files:

                st.markdown(f"**{image_file}**")

                st.image(
                    os.path.join(RESULTS_DIR, image_file),
                    use_container_width=True
                )

    else:

        st.warning(
            "Results folder not found. Create a 'results' folder "
            "and copy your actual evaluation CSV and chart images into it."
        )

    st.markdown("### What do the metrics mean?")

    with st.expander("Accuracy"):
        st.write(
            "The proportion of all predictions that are correct."
        )

    with st.expander("Precision"):
        st.write(
            "Of the samples predicted as a particular class, "
            "how many actually belong to that class."
        )

    with st.expander("Recall"):
        st.write(
            "Of the actual samples belonging to a class, "
            "how many were correctly identified."
        )

    with st.expander("F1-score"):
        st.write(
            "The harmonic mean of precision and recall."
        )


# =====================================================
# 11. DATASET EXPLORER
# =====================================================

elif page == "🗂️ Dataset Explorer":

    section_title(
        "Dataset Explorer 🗂️",
        "Explore the 20 freshness classification categories."
    )

    st.markdown("### Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Total Classes", "20", "Fruit and vegetable classes")

    with col2:
        metric_card("Fresh Classes", "10", "Fresh category labels")

    with col3:
        metric_card("Rotten Classes", "10", "Rotten category labels")

    fruits = [
        "FreshApple", "FreshBanana", "FreshMango",
        "FreshOrange", "FreshStrawberry",
        "RottenApple", "RottenBanana", "RottenMango",
        "RottenOrange", "RottenStrawberry"
    ]

    vegetables = [
        "FreshBellpepper", "FreshCarrot", "FreshCucumber",
        "FreshPotato", "FreshTomato",
        "RottenBellpepper", "RottenCarrot",
        "RottenCucumber", "RottenPotato", "RottenTomato"
    ]

    tab1, tab2 = st.tabs(["🍎 Fruits", "🥕 Vegetables"])

    with tab1:
        st.markdown("### Fruit Categories")

        fruit_df = pd.DataFrame({
            "Category": fruits,
            "Status": [
                "Fresh" if x.startswith("Fresh") else "Rotten"
                for x in fruits
            ]
        })

        st.dataframe(
            fruit_df,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.markdown("### Vegetable Categories")

        vegetable_df = pd.DataFrame({
            "Category": vegetables,
            "Status": [
                "Fresh" if x.startswith("Fresh") else "Rotten"
                for x in vegetables
            ]
        })

        st.dataframe(
            vegetable_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Class Distribution")

    distribution_df = pd.DataFrame({
        "Type": ["Fresh", "Rotten"],
        "Number of Classes": [10, 10]
    })

    st.bar_chart(
        distribution_df.set_index("Type")
    )

    st.caption(
        "This shows the number of categories, not the number of images per category."
    )


# =====================================================
# 12. FOOD TIPS
# =====================================================

elif page == "💡 Food Tips":

    section_title(
        "Food Tips & Awareness 💡",
        "Simple ways to store food and reduce food waste."
    )

    st.markdown("### 🍎 Fruit Storage Tips")

    with st.expander("Apples"):
        st.write(
            "Store apples in the refrigerator to help maintain freshness."
        )

    with st.expander("Bananas"):
        st.write(
            "Keep bananas in a cool, dry place. Refrigeration may darken the peel."
        )

    with st.expander("Mangoes"):
        st.write(
            "Allow unripe mangoes to ripen at room temperature. "
            "Refrigerate ripe mangoes."
        )

    with st.expander("Strawberries"):
        st.write(
            "Refrigerate strawberries and wash them shortly before eating."
        )

    st.markdown("### 🥕 Vegetable Storage Tips")

    with st.expander("Carrots"):
        st.write(
            "Keep carrots refrigerated in a suitable container."
        )

    with st.expander("Potatoes"):
        st.write(
            "Store potatoes in a cool, dark and well-ventilated place."
        )

    with st.expander("Tomatoes"):
        st.write(
            "Allow unripe tomatoes to ripen at room temperature."
        )

    with st.expander("Cucumbers"):
        st.write(
            "Keep cucumbers refrigerated and avoid unnecessary moisture."
        )

    st.markdown("---")

    st.markdown("### 🌱 Reduce Food Waste")

    st.write("""
    - Plan meals before buying groceries.
    - Store food according to its storage needs.
    - Check food regularly.
    - Use older suitable ingredients first.
    - Avoid buying more food than needed.
    """)

    st.warning(
        "Never rely only on appearance or this AI model to determine food safety. "
        "Follow proper food safety guidance."
    )


# =====================================================
# 13. SUGGESTION BOX
# =====================================================

elif page == "📝 Suggestion Box":

    section_title(
        "Suggestion Box 📝",
        "Your feedback helps us improve FoodFresh AI."
    )

    st.markdown("""
    <div class="custom-card">
        <h3 style="color:#941d2c;">We value your feedback!</h3>
        <p>
        Found a bug? Have an idea? Want a new feature?
        Share your thoughts with us.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form"):

        name = st.text_input(
            "Your Name (Optional)",
            max_chars=60
        )

        category = st.selectbox(
            "Feedback Category",
            [
                "Suggestion",
                "Bug Report",
                "Feature Request",
                "General Feedback"
            ]
        )

        rating = st.select_slider(
            "Rate your experience",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x
        )

        message = st.text_area(
            "Your Message",
            placeholder="Write your suggestion or feedback here...",
            max_chars=1000
        )

        submitted = st.form_submit_button(
            "Submit Feedback",
            type="primary",
            use_container_width=True
        )

        if submitted:

            if not message.strip():

                st.error("Please enter a feedback message.")

            else:

                try:

                    save_feedback(
                        name.strip(),
                        category,
                        rating,
                        message.strip()
                    )

                    st.success(
                        "Thank you! Your feedback has been saved successfully."
                    )

                except Exception as error:

                    st.error("Could not save feedback.")
                    st.exception(error)

    st.markdown("---")

    st.markdown("### Previous Feedback")

    if os.path.exists(FEEDBACK_PATH):

        try:

            feedback_df = pd.read_csv(FEEDBACK_PATH)

            st.metric(
                "Total Feedback Submissions",
                len(feedback_df)
            )

            st.dataframe(
                feedback_df.sort_values(
                    by="Date",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        except Exception:
            st.warning("Feedback history could not be displayed.")

    else:

        st.info(
            "No feedback submitted yet. Be the first to share your thoughts!"
        )


# =====================================================
# 14. ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":

    section_title(
        "About FoodFresh AI ℹ️",
        "Understanding the project, its goals and technology."
    )

    st.markdown("### 🎯 Project Objective")

    st.write("""
    The objective of FoodFresh AI is to classify images of
    fruits and vegetables into Fresh and Rotten categories
    using deep learning techniques.
    """)

    st.markdown("### ⚙️ Project Workflow")

    st.markdown("""
    1. Dataset collection.
    2. Image preprocessing and resizing.
    3. Train, validation and test split.
    4. Custom CNN model training.
    5. MobileNetV2 transfer learning.
    6. Model evaluation.
    7. Streamlit application development.
    """)

    st.markdown("### 🧠 Models Used")

    with st.expander("Custom CNN"):
        st.write("""
        A Convolutional Neural Network learns image features
        through convolution and pooling operations.
        """)

    with st.expander("MobileNetV2"):
        st.write("""
        MobileNetV2 is a lightweight deep learning architecture
        designed for efficient image recognition. It uses
        transfer learning in this project.
        """)

    st.markdown("### 💻 Technologies")

    tech_df = pd.DataFrame({
        "Technology": [
            "Python",
            "TensorFlow",
            "Keras",
            "NumPy",
            "Pandas",
            "Pillow",
            "Streamlit"
        ],
        "Purpose": [
            "Programming language",
            "Deep learning framework",
            "Model building and loading",
            "Numerical operations",
            "Data handling",
            "Image processing",
            "Web application interface"
        ]
    })

    st.dataframe(
        tech_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### ⚠️ Limitations")

    st.write("""
    - Predictions depend on image quality.
    - The model only recognizes its trained categories.
    - Confidence scores are not guarantees of correctness.
    - The application is not a food safety inspection tool.
    """)

    st.markdown("---")

    st.markdown(
        "<center>Made with ❤️ using Deep Learning</center>",
        unsafe_allow_html=True
    )


# =====================================================
# 15. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <p style="color:#941d2c;font-weight:600;">
    FoodFresh AI | Deep Learning Project
    </p>
    <p style="color:#888888;font-size:12px;">
    Custom CNN • MobileNetV2 • TensorFlow • Streamlit
    </p>
    </center>
    """,
    unsafe_allow_html=True
)

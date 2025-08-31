# src/ui/ui_utils.py

import base64

import plotly.graph_objects as go
import requests

from src.constants import API_URL, VERSION

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

URGENCY_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

CHART_COLORS = {
    "low": "#FF6B6B",  # Red for low confidence
    "medium": "#FFC107",  # Yellow for medium confidence
    "high": "#28A745",  # Green for high confidence
}

# =============================================================================
# DATA & CLASSIFICATION
# =============================================================================


def get_class_description(class_name):
    """Get comprehensive description for each skin lesion class."""
    descriptions = {
        "akiec": {
            "name": "Actinic Keratoses",
            "description": "Pre-cancerous lesions caused by sun damage",
            "malignancy": "Pre-malignant",
            "urgency": "Medium",
            "icon": "⚠️",
        },
        "bcc": {
            "name": "Basal Cell Carcinoma",
            "description": "Most common type of skin cancer, usually slow-growing",
            "malignancy": "Malignant",
            "urgency": "High",
            "icon": "🚨",
        },
        "bkl": {
            "name": "Benign Keratosis",
            "description": (
                "Non-cancerous skin growth, harmless but may be cosmetically concerning"
            ),
            "malignancy": "Benign",
            "urgency": "Low",
            "icon": "✅",
        },
        "df": {
            "name": "Dermatofibroma",
            "description": "Benign fibrous nodule, typically harmless",
            "malignancy": "Benign",
            "urgency": "Low",
            "icon": "📍",
        },
        "mel": {
            "name": "Melanoma",
            "description": (
                "Most dangerous type of skin cancer, requires immediate attention"
            ),
            "malignancy": "Malignant",
            "urgency": "Critical",
            "icon": "🚨",
        },
        "nv": {
            "name": "Melanocytic Nevi",
            "description": "Common moles, usually benign",
            "malignancy": "Benign",
            "urgency": "Low",
            "icon": "🔍",
        },
        "vasc": {
            "name": "Vascular Lesions",
            "description": "Blood vessel-related skin lesions, typically benign",
            "malignancy": "Benign",
            "urgency": "Low",
            "icon": "❤️",
        },
    }
    return descriptions.get(
        class_name,
        {
            "name": "Unknown",
            "description": "Classification not available",
            "malignancy": "Unknown",
            "urgency": "Unknown",
            "icon": "❓",
        },
    )


# =============================================================================
# API COMMUNICATION
# =============================================================================


def check_api_health():
    """Check if API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_prediction(image_file):
    """Send image to API for prediction with proper error handling."""
    try:
        # Local FastAPI format
        files = {"file": (image_file.name, image_file, image_file.type)}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}

    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# =============================================================================
# FILE HANDLING & VALIDATION
# =============================================================================


def validate_uploaded_file(uploaded_file):
    """Comprehensive file validation."""
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file size (10MB limit)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size exceeds 10MB limit"

    # Check MIME type
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if uploaded_file.type not in allowed_types:
        return False, (
            f"Invalid file type. Only PNG, JPG, JPEG allowed. Got: {uploaded_file.type}"
        )

    # Additional validation: try to open image
    try:
        # image = Image.open(uploaded_file)
        # Reset file pointer after validation
        uploaded_file.seek(0)
        return True, "File is valid"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def load_css_with_background(css_file_path, bg_image_path):
    """Load CSS file and inject background image."""
    try:
        with open(css_file_path, encoding="utf-8") as f:
            css_content = f.read()

        with open(bg_image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()

        css_with_bg = css_content.replace(
            "background: linear-gradient(135deg, #F5DEB3, #DEB887, #D2B48C, #CD853F);",
            f"""background:
                linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                url('data:image/jpeg;base64,{img_data}');""",
        )
        return css_with_bg
    except Exception as e:
        raise Exception(f"Error loading background: {e}") from None


# =============================================================================
# DATA VISUALIZATION
# =============================================================================


def create_confidence_chart(probabilities):
    """Create an interactive confidence chart with proper styling."""
    classes = list(probabilities.keys())
    values = list(probabilities.values())

    # Dynamic color assignment based on confidence levels
    colors = []
    for v in values:
        if v < 0.3:
            colors.append(CHART_COLORS["low"])
        elif v < 0.7:
            colors.append(CHART_COLORS["medium"])
        else:
            colors.append(CHART_COLORS["high"])

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=classes,
                orientation="h",
                marker=dict(color=colors, line=dict(color="white", width=2)),
                text=[f"{v:.1%}" for v in values],
                textposition="inside",
                textfont=dict(color="white", size=12, family="Arial Black"),
            )
        ]
    )

    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Skin Lesion Type",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=False)

    return fig


# =============================================================================
# BUSINESS LOGIC HELPERS
# =============================================================================


def format_prediction_result(result):
    """Format and enrich prediction results with additional metadata."""
    if "error" in result:
        return result

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    class_info = get_class_description(predicted_class)

    return {
        **result,
        "class_info": class_info,
        "urgency_color": URGENCY_COLORS.get(class_info.get("urgency", "Unknown"), "⚪"),
        "is_high_risk": class_info.get("urgency") in ["High", "Critical"],
        "formatted_confidence": f"{confidence:.1%}",
    }


def get_sidebar_info():
    """Get standardized sidebar information."""
    return {
        "about_text": """
        This AI system uses a ResNet50 deep learning model
        to analyze skin lesion images and classify them into
        7 categories of skin conditions.

        **⚠️ Important:** This tool is for educational purposes only.
        Always seek professional medical advice for diagnosis or treatment.
        """,
        "technical_details": """
        Dataset: HAM10000 (~10,000 dermatology images)

        Architecture: ResNet50 CNN
        Regularization: Custom dropout layers
        Optimization: Adaptive learning rate scheduling

        Results:
        • Training Accuracy: ~99%
        • Validation Accuracy: ~88.12%
        • Test Accuracy: ~87.97%
        • Recall: 60%+ across all classes

        Limitations:
        • Overfitting: 99% train vs. 88% validation/test.
        • Weak generalization on minority classes.
        • Melanoma detection still needs improvement.
        • Augmentation does not cover real-world variation.

        Future Improvements:
        • Add stronger regularization.
        • Improve class balance.
        • Use richer augmentation.
        • Boost generalization on critical classes.
        """,
        "social_links": """
        💼 [LinkedIn](https://linkedin.com/in/lisekarimi)
        🐱 [GitHub](https://github.com/lisekarimi/skindx)
        📊 [Kaggle: Model Training Notebook](https://www.kaggle.com/code/lizk75/skin-cancer-resnet-balanced-87-acc)
        """,
        "version": VERSION,  # Update this as needed
        "changelog_url": "https://github.com/lisekarimi/skindx/blob/main/CHANGELOG.md",
    }


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================


def handle_prediction_error(error_message):
    """Standardized error handling for predictions."""
    error_types = {
        "timeout": "⏱️ Analysis timed out. Please try with a smaller image.",
        "connection": "🔌 Cannot connect to AI service. Please try again later.",
        "file": "📁 File processing error. Please check your image format.",
        "api": "⚠️ AI service error. Please contact support if this persists.",
    }

    # Determine error type based on message content
    for key, friendly_message in error_types.items():
        if key.lower() in error_message.lower():
            return friendly_message

    return f"❌ {error_message}"

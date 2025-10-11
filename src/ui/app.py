# src/ui/app.py
import os

import streamlit as st
from PIL import Image

from src.constants import LESION_CATEGORIES
from src.ui.ui_utils import (
    create_confidence_chart,
    format_prediction_result,
    get_prediction,
    get_sidebar_info,
    handle_prediction_error,
    load_css_with_background,
    validate_uploaded_file,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SKINDX - AI Skin Analysis", page_icon="🔬", layout="wide"
)

# =============================================================================
# STYLING & ASSETS
# =============================================================================


def load_custom_styles():
    """Load custom CSS styling."""
    css_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "styles.css"
    )
    bg_image = os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "static", "bg.jpg"
    )

    try:
        css_with_bg = load_css_with_background(css_file, bg_image)
        st.markdown(f"<style>{css_with_bg}</style>", unsafe_allow_html=True)

        # Hide file uploader info box
        st.markdown(
            """
        <style>
        .uploadedFile {
            display: none !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error loading styles: {e}")


# =============================================================================
# UI COMPONENTS
# =============================================================================


def render_header():
    """Render main header section."""
    st.markdown('<h1 class="main-header">🔬 SKINDX</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-Powered Skin Lesion Analysis</p>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with information and navigation."""
    sidebar_info = get_sidebar_info()

    with st.sidebar:
        st.markdown("### 📋 About SKINDX")
        st.info(sidebar_info["about_text"])

        with st.expander("🔬 Technical Details"):
            st.text(sidebar_info["technical_details"])

        st.markdown("### Supported Lesion Types")
        for category, lesions in LESION_CATEGORIES.items():
            st.markdown(category)
            for code, description in lesions:
                st.write(f"• {description} ({code.upper()})")

        # Social links
        st.markdown("---")
        st.markdown("### Connect with Us")
        st.markdown(sidebar_info["social_links"])

        # Version and Changelog
        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown(
            f"🔖 **Version:** `{sidebar_info['version']}` | "
            f"📝 [Changelog]({sidebar_info['changelog_url']}) | "
            f"📚 [Wiki]({sidebar_info['doc_url']})"
        )


def render_prediction_results(result):
    """Render prediction results in a formatted way."""
    formatted_result = format_prediction_result(result)

    if "error" in formatted_result:
        error_message = handle_prediction_error(formatted_result["error"])
        st.error(error_message)
        return

    # Extract formatted data
    class_info = formatted_result["class_info"]
    urgency_color = formatted_result["urgency_color"]
    confidence = formatted_result["formatted_confidence"]

    # Display results
    st.success("Analysis Complete!")

    # Main result card
    st.markdown(
        f"## {class_info.get('icon', '🔬')} {class_info.get('name', 'Unknown')}"
    )
    st.markdown(f"**Confidence:** {confidence}")
    st.markdown(
        f"**Description:** {class_info.get('description', 'No description available')}"
    )
    st.markdown(f"**Malignancy:** {class_info.get('malignancy', 'Unknown')}")

    # Priority indicator with color
    urgency = class_info.get("urgency", "Unknown")
    st.markdown(f"**Priority Level:** {urgency_color} {urgency}")

    # High-risk warning
    if formatted_result.get("is_high_risk"):
        st.warning("⚠️ **High Priority:** Consider consulting a dermatologist soon.")

    # Confidence breakdown chart
    st.markdown("### 📊 Confidence Breakdown")
    chart = create_confidence_chart(result["all_probabilities"])
    st.plotly_chart(chart, use_container_width=True)

    # Medical disclaimer
    st.warning(
        "⚠️ **Medical Disclaimer:** This AI analysis is for educational purposes only. "
        "Always consult healthcare professionals for medical advice."
    )


def render_upload_interface():
    """Render the main upload and prediction interface."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 📸 Upload Skin Lesion Image")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear image of the skin lesion",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            # Validate file
            is_valid, validation_message = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {validation_message}")
            else:
                # Display image only if valid
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict button
            if st.button(
                "🔍 Analyze Skin Lesion", use_container_width=True, type="primary"
            ):
                with st.spinner("🤖 Analyzing your image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Get prediction
                    result = get_prediction(uploaded_file)

                    # Render results
                    render_prediction_results(result)


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Return the main application entry point."""
    # Load custom styles
    load_custom_styles()

    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Render main interface
    render_upload_interface()

    st.markdown(
        """
        <a href="/docs/" class="floating-chat-btn" target="_blank">
            💬 Chat with our AI Assistant
        </a>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

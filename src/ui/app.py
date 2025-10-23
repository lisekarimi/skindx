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
    page_title="SKINDX - AI Skin Analysis",
    page_icon="assets/static/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
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


def inject_og_meta_tags():
    """Inject Open Graph meta tags for social media sharing."""
    # Convert to URL path (assuming the app serves static files from assets/static/)
    og_image_url = "https://skindx.lisekarimi.com/assets/static/og-img.png"

    # Get the current URL (you might want to customize this based on your deployment)
    current_url = "https://skindx.lisekarimi.com"  # Update this with your actual domain

    description = (
        "AI-powered skin lesion analysis using deep learning. "
        "Upload an image to get instant analysis of skin conditions "
        "with confidence scores and medical insights."
    )
    keywords = (
        "AI, skin analysis, dermatology, machine learning, medical AI, "
        "skin lesions, melanoma detection"
    )

    og_meta_tags = f"""
    <head>
        <!-- Open Graph Meta Tags -->
        <meta property="og:title" content="SKINDX - AI Skin Analysis">
        <meta property="og:description" content="{description}">
        <meta property="og:image" content="{current_url}{og_image_url}">
        <meta property="og:url" content="{current_url}">
        <meta property="og:type" content="website">
        <meta property="og:site_name" content="SKINDX">

        <!-- Twitter Card Meta Tags -->
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="SKINDX - AI Skin Analysis">
        <meta name="twitter:description" content="{description}">
        <meta name="twitter:image" content="{current_url}{og_image_url}">

        <!-- Additional Meta Tags -->
        <meta name="description" content="{description}">
        <meta name="keywords" content="{keywords}">
        <meta name="author" content="Lise Karimi">
    </head>
    """

    st.markdown(og_meta_tags, unsafe_allow_html=True)


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
            f"📚 [Docs]({sidebar_info['doc_url']})"
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


def render_portfolio_section():
    """Render portfolio call-to-action section."""
    st.markdown("---")
    st.markdown(
        """
        <div class="portfolio-section">
            <div class="portfolio-content">
                <h2 class="portfolio-title">🌟 Explore More Projects</h2>
                <p class="portfolio-description">
                    Inspired by this AI solution? Explore more ML and AI projects. 🚀
                </p>
                <a href="https://lisekarimi.com"
                   target="_blank" class="portfolio-btn">
                    View My Portfolio
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Return the main application entry point."""
    # Inject Open Graph meta tags for social media sharing
    inject_og_meta_tags()

    # Load custom styles
    load_custom_styles()

    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Render main interface
    render_upload_interface()

    # Render portfolio section
    render_portfolio_section()

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

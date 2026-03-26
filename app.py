import streamlit as st
from predict_utils import predict_two_stage

st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

st.title("AI Cyberbullying Detection System")
st.write("Supports text, emojis, and context-aware analysis")

# Inputs
comment = st.text_area("Enter Comment")
caption = st.text_input("Caption (optional)")
replies = st.text_input("Prior Replies (comma separated)")

if st.button("Analyze"):

    if not comment.strip():
        st.warning("Please enter a comment")
    else:
        reply_list = [r.strip() for r in replies.split(",")] if replies else None

        result = predict_two_stage(
            comment,
            parent_caption=caption if caption else None,
            prior_replies=reply_list
        )

        st.subheader("Prediction Result")

        # Display results nicely
        st.write("Binary:", result["binary_label"])
        st.write("Category:", result["category"])

        # Visual feedback
        if result["binary_label"] == "threat":
            st.error("🚨 Threat Detected")
            st.progress(0.95)

        elif result["category"] == "harassment":
            st.warning("⚠️ Harassment Detected")
            st.progress(0.75)

        elif result["category"] == "sexual":
            st.warning("⚠️ Sexual Content Detected")
            st.progress(0.70)

        else:
            st.success("✅ Safe / Neutral Content")
            st.progress(0.30)
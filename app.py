# ===============================================================
# veritas_churn_dashboard.py — Veritas Bank Churn Analysis Assistant
# ===============================================================
import streamlit as st
import os
import fitz
import pandas as pd
from PIL import Image
from docx import Document
from gemini import chat_with_knowledge_base, generate_comprehensive_report
import io
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Veritas Bank Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# Knowledge Base Loader
# ---------------------------------------------------------------
@st.cache_data
def load_knowledge_base():
    """Load knowledge base from Veritas Bank churn analysis docx and Excel files"""
    try:
        doc = Document("attached_assets/VeritasBank_Churn_Knowledge_Base.docx")
        knowledge_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Load data from Excel
        customer_info = pd.read_excel("attached_assets/CustomerInfo.xlsx")
        account_info = pd.read_excel("attached_assets/AccountInfo.xlsx")
        
        merged = pd.merge(customer_info, account_info, on="CustomerId", how="inner")

        churn_rate = merged["Exited"].mean() * 100
        active_members = merged["ActiveMember"].sum()
        inactive_members = len(merged) - active_members
        avg_credit = merged["CreditScore"].mean()
        avg_balance = merged["Balance"].mean()
        avg_age = merged["Age"].mean()
        countries = merged["Country"].value_counts().to_string()
        genders = merged["Gender"].value_counts().to_string()

        excel_summary = f"""

ADDITIONAL DATASET SUMMARY – VERITAS BANK CUSTOMER PORTFOLIO:

CUSTOMER OVERVIEW:
- Total Customers: {len(merged)}
- Active Customers: {active_members}
- Inactive Customers: {inactive_members}
- Average Age: {avg_age:.1f} years
- Gender Distribution:
{genders}
- Country Distribution:
{countries}

ACCOUNT INFORMATION:
- Average Credit Score: {avg_credit:.1f}
- Average Balance: £{avg_balance:,.2f}
- Churn Rate: {churn_rate:.2f}%

PRODUCT & ENGAGEMENT:
- Average Products Held: {merged['Products'].mean():.2f}
- Active Members (%): {(merged['ActiveMember'].mean() * 100):.2f}%

CREDIT TIER DISTRIBUTION:
{merged['CreditScore'].apply(lambda x: 'Low' if x < 580 else 'Moderate' if x < 700 else 'High').value_counts().to_string()}
"""
        
        return knowledge_text + excel_summary
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        return ""

# ---------------------------------------------------------------
# Optional: Convert PDF Dashboard to Image (if included)
# ---------------------------------------------------------------
@st.cache_data
def convert_pdf_to_image():
    """Convert Veritas Bank PDF dashboard to image"""
    try:
        pdf_path = "attached_assets/VeritasBank_Churn_Report.pdf"
        with fitz.open(pdf_path) as pdf_document:
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None

# ---------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------
def display_header():
    """Display header with Veritas Bank branding"""
    st.markdown("""
        <h1 style='text-align: center; color: #fe7a28; font-size: 48px;'>🏦 Veritas Bank</h1>
        <p style='text-align: center; font-size: 24px; color: #444;'>Customer Churn Analysis Dashboard & AI Assistant</p>
        <hr style='margin-bottom: 20px;'>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Main App Logic
# ---------------------------------------------------------------
def main():
    display_header()
    
    knowledge_base = load_knowledge_base()
    
    main_col, chat_col = st.columns([2, 1])
    
    # ----------------- MAIN COLUMN -----------------
    with main_col:
        st.markdown("### 📊 Interactive Power BI Dashboard")

        # Embed Power BI live churn report
        st.markdown(
            """
            <iframe title="Veritas Bank Churn Dashboard" width="100%" height="700"
            src="https://app.powerbi.com/view?r=eyJrIjoiMjc3ZGMwZTUtYzZlMy00NzA4LWFhNDItYzdmMDI4MjIyNDUwIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9"
            frameborder="0" allowFullScreen="true"></iframe>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📄 Generate Comprehensive Churn Report", type="primary"):
            with st.spinner("Generating Veritas Bank churn insights report..."):
                if not os.environ.get("GEMINI_API_KEY"):
                    st.error("⚠️ GEMINI_API_KEY not set. Please add API key in your .env file.")
                else:
                    report = generate_comprehensive_report(knowledge_base)
                    st.markdown("### 📋 Comprehensive Business Insights Report")
                    st.markdown(report)
    
    # ----------------- CHAT COLUMN -----------------
    with chat_col:
        st.markdown("### 🤖 AI Assistant")
        st.markdown("*Ask questions about customers, churn, engagement, or risk profiles*")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about churn rate, customer segments, or balance insights..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if not os.environ.get("GEMINI_API_KEY"):
                response = "⚠️ Please set your GEMINI_API_KEY to use the chatbot."
            else:
                with st.spinner("Analyzing churn data..."):
                    response = chat_with_knowledge_base(prompt, knowledge_base, st.session_state.chat_history)
                    st.session_state.chat_history.append(f"User: {prompt}")
                    st.session_state.chat_history.append(f"Assistant: {response}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        with st.expander("💡 Sample Questions"):
            st.markdown("""
            - What is the current churn rate at Veritas Bank?
            - Which country has the highest churn rate?
            - How does credit score affect churn likelihood?
            - What is the average balance of churned customers?
            - Are inactive members more likely to churn?
            - Which customer age group has the highest attrition?
            - What are the top retention recommendations?
            - Show me the churn distribution by engagement level.
            """)

# ---------------------------------------------------------------
# Run the Streamlit App
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()

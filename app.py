import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="Shopper Spectrum Portal", layout="wide")

# Set matplotlib style sheets for dark backgrounds
plt.style.use('dark_background')

# 2. Injecting Premium Custom Styling (Gradients, Colors, Cards)
st.markdown("""
    <style>
    /* Gradient Background for the App Body */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc !important;
    }
    
    /* Executive Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #f3f4f6 !important;
    }
    
    /* Premium Title Header Banner */
    .main-title-banner {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    .main-title-banner h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Glassmorphism Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Result Display Cards */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .res-high { background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; color: #34d399 !important; }
    .res-regular { background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; color: #60a5fa !important; }
    .res-occasional { background: rgba(245, 158, 11, 0.2); border: 2px solid #f59e0b; color: #fbbf24 !important; }
    .res-atrisk { background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; color: #f87171 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. Cache Data Sources Safely
@st.cache_data
def load_data():
    try:
        clean_df = pd.read_csv('online_retail_cleaned_small.csv')
        clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])
        rfm_df = pd.read_csv('customer_segments.csv')
        similarity_df = pd.read_csv('product_similarity_small.csv', index_col=0)
        return clean_df, rfm_df, similarity_df
    except Exception:
        return None, None, None

df_clean, rfm, similarity_df = load_data()

# 4. Premium Sidebar Panel Layout
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h2 style='color: #ec4899 !important; font-weight: 800; margin-bottom: 0;'>📊 CONTROL CORE</h2>
        <hr style='border-color: rgba(255,255,255,0.15); margin-top: 10px; margin-bottom: 20px;'>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate System Modules:", [
    "🏢 Core Overview Panel", 
    "📈 Business Intelligence Charts",
    "👥 Customer Segmentation Engine", 
    "📦 Product Recommendation Engine"
])

# ==========================================
# MODULE 1: OVERVIEW PANEL
# ==========================================
if page == "🏢 Core Overview Panel":
    st.markdown("""
        <div class='main-title-banner'>
            <h1>✨ SHOPPER SPECTRUM</h1>
            <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;'>Enterprise Customer Intelligence & Engine Analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Active Operational Subsystems")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <h3 style='color: #8b5cf6 !important; margin-top:0;'>👥 Behavior Clustering</h3>
                <p>Processes customer coordinates across calculated Recency, Frequency, and Monetary spaces via our trained K-Means optimization script pipeline.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <h3 style='color: #ec4899 !important; margin-top:0;'>📦 Cross-Sell Matrices</h3>
                <p>Utilizes user-item sparse arrays mapping dynamic cosine metrics vectors to extract complementary recommendation assets from inventory pools instantly.</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# MODULE 2: DYNAMIC GRAPH VISUALIZATIONS
# ==========================================
elif page == "📈 Business Intelligence Charts":
    st.markdown("<div class='main-title-banner'><h1>📈 Executive Data Analytics</h1></div>", unsafe_allow_html=True)
    
    if df_clean is not None and rfm is not None:
        # Chart Row 1: Distribution of Segments & Total Revenue Trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='feature-card'><h4>📊 Customer Segment Volume</h4>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            segment_counts = rfm['Segment'].value_counts()
            sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='magma', ax=ax1)
            plt.xticks(rotation=15)
            ax1.set_ylabel("Total Customers")
            ax1.patch.set_alpha(0.0)
            fig1.patch.set_alpha(0.0)
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='feature-card'><h4>💰 Sales Revenue Trend Line</h4>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            daily_revenue = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalSpend'].sum().reset_index()
            daily_revenue['InvoiceDate'] = daily_revenue['InvoiceDate'].astype(str)
            sns.lineplot(data=daily_revenue, x='InvoiceDate', y='TotalSpend', marker='o', color='#ec4899', ax=ax2)
            plt.xticks(rotation=45)
            ax2.patch.set_alpha(0.0)
            fig2.patch.set_alpha(0.0)
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart Row 2: Top Selling Products Table & Scatter Metrics
        st.markdown("<div class='feature-card'><h4>🛒 Top 10 High Demand Products by Volume</h4>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax3)
        ax3.patch.set_alpha(0.0)
        fig3.patch.set_alpha(0.0)
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Please ensure 'online_retail_cleaned.csv' and 'customer_segments.csv' are present in your workspace.")

# ==========================================
# MODULE 3: SEGMENTATION ENGINE
# ==========================================
elif page == "👥 Customer Segmentation Engine":
    st.markdown("<div class='main-title-banner'><h1>👥 Customer Behavioral Classifier</h1></div>", unsafe_allow_html=True)
    
    st.markdown("### 📥 Metric Coordinates Input")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days Since Last Order Window)", min_value=1, max_value=365, value=45)
    with col2:
        frequency = st.number_input("Frequency (Total Invoiced Interactivity Count)", min_value=1, max_value=500, value=12)
    with col3:
        monetary = st.number_input("Monetary (Gross Operational Valuation Lifetime)", min_value=1.0, max_value=1000000.0, value=2450.0)
        
    if st.button("🚀 Process Model Execution", use_container_width=True):
        if monetary >= 5000 or frequency >= 50:
            seg_class, seg_label = "res-high", "🏆 HIGH-VALUE STRATEGIC BUYER"
        elif recency >= 180:
            seg_class, seg_label = "res-atrisk", "⚠️ HIGH CHURN AT-RISK ACCOUNT"
        elif frequency <= 2:
            seg_class, seg_label = "res-occasional", "🪕 OCCASIONAL CASUAL SHOPPER"
        else:
            seg_class, seg_label = "res-regular", "🛡️ STABLE REGULAR ACCOUNT"
            
        st.markdown(f"<div class='result-box {seg_class}'>IDENTIFIED SPECTRUM STATUS: {seg_label}</div>", unsafe_allow_html=True)

# ==========================================
# MODULE 4: PRODUCT RECOMMENDATIONS
# ==========================================
elif page == "📦 Product Recommendation Engine":
    st.markdown("<div class='main-title-banner'><h1>📦 Collaborative Matrix Recommendations</h1></div>", unsafe_allow_html=True)
    
    if similarity_df is not None:
        product_list = sorted(similarity_df.index.tolist())
        selected_product = st.selectbox("Search Target SKU Inventory Listing Description:", product_list)
        
        if st.button("🔍 Generate Vector Affinities", use_container_width=True):
            if selected_product in similarity_df.index:
                recommendations = similarity_df[selected_product].sort_values(ascending=False).iloc[1:6]
                st.markdown("### 🌟 Target Inventory Matching Matrix Output")
                for idx, (item, score) in enumerate(recommendations.items(), 1):
                    st.markdown(f"""
                        <div class='feature-card' style='padding: 15px; margin: 8px 0;'>
                            <span style='color: #ec4899 !important; font-weight: bold;'>SKU rank 0{idx}:</span> 
                            <span style='margin-left: 10px;'>{item}</span>
                            <span style='float: right; color: #8b5cf6 !important; font-weight: bold;'>Match Score: {score:.4f}</span>
                        </div>
                    """, unsafe_allow_html=True)
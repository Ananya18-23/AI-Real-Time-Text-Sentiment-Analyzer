import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import time

# 1. Page Configuration (Professional Wide Layout)
st.set_page_config(
    page_title="AI Sentiment Insights Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. NLTK Data Download (Automated & Cached)
@st.cache_resource
def load_vader():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")
        return None

sia = load_vader()

# --- SIDEBAR SECTION (WITH DYNAMIC EDITOR) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 AI Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 🛠️ SECTION 1: DYNAMIC PROFILE EDITOR (Yahan se aap change kar sakte hain)
    with st.expander("📝 Edit Profile Settings"):
        st.write("Apni details yahan change karein:")
        # Default value "Amit Kumar" rakhi hai, aap web page par isko badal sakte hain
        dev_name = st.text_input("Aapka Naam:", value="Amit Kumar")
        dev_role = st.text_input("Aapka Role:", value="GenAI & NLP Developer")
        github_url = st.text_input("GitHub Link:", value="https://github.com/")
        linkedin_url = st.text_input("LinkedIn Link:", value="https://linkedin.com/")
    
    st.markdown("---")
    
    # 🧑‍💻 SECTION 2: DEVELOPER PROFILE CARD (Jo real-time update hoga)
    st.markdown("### 🧑‍💻 Developer Profile")
    
    # Yeh inputs upar wale text boxes se live data uthayenge
    st.markdown(f"#### **{dev_name}**") 
    st.caption(f"🚀 {dev_role}")
    
    # Interactive Skill Badges
    st.markdown("""
    <div style='display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 10px;'>
        <span style='background-color: #0077B5; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>Python</span>
        <span style='background-color: #FF4B4B; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>Streamlit</span>
        <span style='background-color: #23d160; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>NLP</span>
        <span style='background-color: #7289da; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic Social Links
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown(f"[🌐 GitHub]({github_url})")
    with col_link2:
        st.markdown(f"[💼 LinkedIn]({linkedin_url})")
        
    st.markdown("---")
    
    # Feature 3: Real-time Text Analytics Status
    st.markdown("### 📊 Live Text Metrics")
    user_input_text = st.session_state.get('text_input_area', "")
    word_count = len(user_input_text.split()) if user_input_text.strip() != "" else 0
    char_count = len(user_input_text)
    
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        st.metric(label="Words", value=word_count)
    with col_side2:
        st.metric(label="Characters", value=char_count)
        
    st.markdown("---")
    
    # Feature 4: Model Configuration
    st.markdown("### ⚙️ Engine Settings")
    selected_model = st.selectbox("NLP Model", ["NLTK-VADER (Rule-Based)", "BERT-Transformer (Enterprise - Locked)"])
    confidence_threshold = st.slider("Neutral Zone Threshold", 0.01, 0.10, 0.05, step=0.01)
    
    st.markdown("---")
    st.caption("🟢 System Status: Healthy & Operational")
    
    
# --- MAIN APPLICATION UI ---
st.title("🧠 AI Real-Time Text Sentiment Analyzer")
st.markdown("""
    Welcome to the advanced sentiment analyzer. This production-grade tool uses 
    **Natural Language Processing (NLP)** to break down text emotion into precise metrics.
""")

# Input box for user text (Linked with session state for sidebar metrics)
user_text = st.text_area(
    "📝 Enter your English text, review, or tweet below:", 
    height=150, 
    placeholder="Type something here... e.g., The customer service was absolutely fantastic, but the delivery was a bit slow.",
    key="text_input_area" # This key connects it to the sidebar live metrics
)

st.markdown("---")

# 5. Analysis Logic
if st.button("🔥 Run Advanced Analysis", use_container_width=True):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    elif sia is None:
        st.error("🚨 Sentiment analyzer model could not be initialized.")
    else:
        # Start timer for system metrics
        start_time = time.time()
        
        with st.spinner("Analyzing linguistic patterns..."):
            # Model prediction
            scores = sia.polarity_scores(user_text)
            
            # Extract scores
            compound = scores['compound']
            pos_score = scores['pos']
            neu_score = scores['neu']
            neg_score = scores['neg']
            
            # Determine overall sentiment verdict based on sidebar threshold
            if compound >= confidence_threshold:
                verdict = "POSITIVE"
                verdict_emoji = "🟢 😊"
                color_func = st.success
            elif compound <= -confidence_threshold:
                verdict = "NEGATIVE"
                verdict_emoji = "🔴 😡"
                color_func = st.error
            else:
                verdict = "NEUTRAL"
                verdict_emoji = "🟡 😐"
                color_func = st.info
                
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # --- DISPLAY RESULTS IN LAYOUT ---
            st.subheader("📊 Analysis Verdict")
            color_func(f"### Overall Sentiment: **{verdict}** {verdict_emoji}")
            
            # Metric Columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Overall Sentiment Score", value=f"{compound:.2f}")
            with col2:
                st.metric(label="Positive Intensity", value=f"{pos_score * 100:.1f}%")
            with col3:
                st.metric(label="Neutral Intensity", value=f"{neu_score * 100:.1f}%")
            with col4:
                st.metric(label="Negative Intensity", value=f"{neg_score * 100:.1f}%")
                
            st.markdown("### 📈 Visual Breakdown")
            
            # Creating two columns for advanced visual representation
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.write("**Linguistic Breakdown (Data Table)**")
                # Pandas DataFrame for professional data formatting
                df_scores = pd.DataFrame({
                    "Metric": ["Positive", "Neutral", "Negative"],
                    "Score Percentage (%)": [pos_score * 100, neu_score * 100, neg_score * 100]
                })
                st.dataframe(df_scores, hide_index=True, use_container_width=True)
                
                # Bonus Feature: Export Report Button
                csv = df_scores.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sentiment Report (CSV)",
                    data=csv,
                    file_name='sentiment_analysis_report.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
            with vis_col2:
                st.write("**Visual Chart**")
                # Plotly Chart for interactive graphs
                fig = px.bar(
                    df_scores, 
                    x="Metric", 
                    y="Score Percentage (%)", 
                    color="Metric",
                    color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
                )
                fig.update_layout(showlegend=False, height=220, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
            # Sidebar dynamic update after execution
            with st.sidebar:
                st.markdown("---")
                st.markdown("### ⚡ System Performance")
                st.text(f"Latency: {execution_time*1000:.2f} ms")
import streamlit as st
import numpy as np
import gensim
from joblib import load
import time
from datetime import datetime

# ---- Configuration and Page Setup ----
st.set_page_config(
    page_title="K-Mail Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for styling ----
st.markdown("""
<style>
    /* Global settings */
    body {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #e0e0e0;
        background-color: #121212;
    }
    
    /* Sidebar styling - Gemini-like gradient */
    .css-1d391kg {
        background: linear-gradient(180deg, #121212 0%, #1a1a2e 100%);
    }
    
    /* Main container - Gemini-like gradient */
    .main .block-container {
        background: linear-gradient(135deg, #121212 0%, #1a1a2e 50%, #291b51 100%);
        padding: 2rem;
        border-radius: 1rem;
    }
    
    /* Headers - Gemini-like colors */
    h1, h2, h3, h4, h5, h6 {
        color: #8b5cf6;
        font-weight: 600;
    }
    
    /* Buttons - Gemini-like color scheme */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7e22ce 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(126, 34, 206, 0.4);
    }
    
    /* Feature Cards - Gemini-like color scheme */
    .feature-card {
        background: linear-gradient(145deg, #1a1a2e, #291b51);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        border-left: 4px solid #4f46e5;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #7e22ce;
    
    }
    
    /* Email List - Gemini-like colors */
    .email-list {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }
    .email-list:hover {
        background-color: #291b51;
        border-left: 3px solid #4f46e5;
    }
    .email-list.selected {
        background-color: #291b51;
        border-left: 3px solid #4f46e5;
    }
    
    /* Email Detail View */
    .email-detail {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #333;
    }
    .email-header {
        border-bottom: 1px solid #333;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .email-body {
        padding: 1rem 0;
        color: #ccc;
    }
    .spam-badge {
        background-color: #f44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .safe-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Logo - Gemini-like colors */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 2rem;
    }
    .logo-text {
        color: #4f46e5;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Landing Hero - Gemini-like gradient */
    .hero-container {
        text-align: center;
        
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(126, 34, 206, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 3rem;
    }
    .hero-title {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #4f46e5 0%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #ccc;
        margin-bottom: 2rem;
    }
    
    /* User Input - Gemini-like focus color */
    .stTextArea > div > div {
        background-color: #1a1a2e;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 10px;
    }
    .stTextArea > div > div:focus-within {
        border-color: #4f46e5;
    }
    
    /* Widget labels */
    .css-16idsys p {
        color: #e0e0e0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation for unread emails */
    @keyframes pulse {
        0% {opacity: 1;}
        50% {opacity: 0.7;}
        100% {opacity: 1;}
    }
    .unread {
        animation: pulse 2s infinite;
        font-weight: bold;
    }
    
    /* Statistics container - Gemini-like gradient */
    .stats-container {
        background: linear-gradient(145deg, #1a1a2e, #291b51);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4f46e5;
    }
    .stat-label {
        color: #ccc;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Session State Initialization ----
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'inbox' not in st.session_state:
    st.session_state.inbox = []
if 'spam' not in st.session_state:
    st.session_state.spam = []
if 'sent' not in st.session_state:
    st.session_state.sent = []
if 'pointer' not in st.session_state:
    st.session_state.pointer = 0
if 'selected' not in st.session_state:
    st.session_state.selected = ('inbox', None)
if 'folder' not in st.session_state:
    st.session_state.folder = 'Inbox'
if 'viewed_emails' not in st.session_state:
    st.session_state.viewed_emails = set()

# ---- Load Models (cached resource) ----
@st.cache_resource
def load_models():
    lg = load("spam_detec_model")
    w2v = gensim.models.Word2Vec.load("spam_detec_w2v")
    return lg, w2v

model, w2v_model = load_models()

# ---- Prediction Function ----
def pred(text, w2v, lg):
    tokens = gensim.utils.simple_preprocess(text)
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    vec = np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)
    return int(lg.predict(vec.reshape(1, -1))[0])

# ---- Predefined Emails ----
PREDEFINED_EMAILS = [
    {
        "subject": "Win a free iPhone 15!", 
        "body": "Congratulations! You've been selected to receive a free iPhone 15. Click here to claim your prize now. Limited time offer!", 
        "label": 1,
        "date": "2025-04-29 08:15"
    },
    {
        "subject": "Team Meeting Agenda", 
        "body": "Dear team,\n\nPlease find attached the agenda for tomorrow's meeting. We'll be discussing Q2 projections and the new marketing strategy.\n\nBest regards,\nAlex", 
        "label": 0,
        "date": "2025-04-29 09:22"
    },
    {
        "subject": "URGENT: Your account has been compromised", 
        "body": "We detected unusual activity on your account. Your password needs to be reset immediately. Click this link to secure your account now!", 
        "label": 1,
        "date": "2025-04-29 10:07"
    },
    {
        "subject": "Project Update - Phase 2", 
        "body": "Hi,\n\nI wanted to let you know that we've completed Phase 1 of the project and are now moving to Phase 2. The initial results look promising.\n\nLet me know if you have any questions.\n\nRegards,\nSamantha", 
        "label": 0,
        "date": "2025-04-29 11:45"
    },
    {
        "subject": "You've won $5,000,000 in the lottery!", 
        "body": "Dear Lucky Winner,\n\nCongratulations! Your email address has won $5,000,000 in our international lottery. To claim your prize, please send us your personal details and a small processing fee.", 
        "label": 1,
        "date": "2025-04-29 12:33"
    },
    {
        "subject": "Quarterly Report - Q1 2025", 
        "body": "Please find attached the quarterly report for Q1 2025. The numbers are looking good, with a 15% increase in revenue compared to last quarter.\n\nThanks,\nFinance Team", 
        "label": 0,
        "date": "2025-04-29 13:18"
    },
    {
        "subject": "LAST CHANCE: 90% OFF Designer Watches", 
        "body": "INCREDIBLE SALE!!! Genuine Rolex, Omega, and Tag Heuer watches at 90% discount! Limited stock available. Buy now and save thousands!!!", 
        "label": 1,
        "date": "2025-04-29 14:05"
    },
    {
        "subject": "Following up on our conversation", 
        "body": "Hi there,\n\nI'm just following up on our conversation from last week about the new client proposal. Have you had a chance to review the documents I sent?\n\nThanks,\nJamie", 
        "label": 0,
        "date": "2025-04-29 15:39"
    },
    {
        "subject": "Medication at LOW PRICES - No Prescription Needed!", 
        "body": "Buy premium medications without prescription! We ship worldwide. Discreet packaging guaranteed. LOWEST PRICES ONLINE!!!", 
        "label": 1,
        "date": "2025-04-29 16:27"
    },
    {
        "subject": "Reminder: Benefits Enrollment Deadline", 
        "body": "This is a friendly reminder that the annual benefits enrollment period ends this Friday. Please complete your selections by then to ensure coverage for the upcoming year.", 
        "label": 0,
        "date": "2025-04-29 17:10"
    }
]

# ---- Helper Functions ----
def format_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    return date_obj.strftime("%b %d, %H:%M")

def get_email_preview(body, max_length=50):
    if len(body) > max_length:
        return body[:max_length] + "..."
    return body

# ---- Home Page (Landing) ----
def render_home():
    # Logo and header
    st.markdown('<div class="logo-container"><span class="logo-text">K-Mail</span><span style="color:#e0e0e0;">Spam Detector</span></div>', unsafe_allow_html=True)
    
    
    
    # Hero section
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Intelligent Spam Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Keep your inbox clean with our AI-powered spam detection system</p>', unsafe_allow_html=True)
    
    if st.button('Try it now!', key='to_app'):
        st.session_state.page = 'app'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature cards
    st.subheader("Advanced Features")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>Machine Learning Powered</h3>
            <p>Our system uses a logistic regression model trained on diverse datasets from three sources to accurately classify messages.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Real-Time Detection</h3>
            <p>Instant analysis of incoming messages with real-time classification to protect you from the latest spam techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3>Semantic Analysis</h3>
            <p>Text data is vectorized using Word2Vec to capture semantic word relationships for improved accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <h3>Robust Protection</h3>
            <p>Our system effectively filters out spam while ensuring important messages reach your inbox.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💻</div>
            <h3>User-Friendly Interface</h3>
            <p>Intuitive web application that makes spam detection accessible and efficient for all users.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Detailed Analytics</h3>
            <p>Get insights into your email traffic with statistics on spam detection and inbox management.</p>
        </div>
        """, unsafe_allow_html=True)

# ---- Main App Page ----
def render_app():
    # Logo in sidebar
    with st.sidebar:
        st.markdown('<div class="logo-container"><span class="logo-text">K-Mail</span><span style="color:#e0e0e0;">Spam Detector</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Folder navigation in sidebar
        st.subheader("Folders")
        folders = ['Inbox', 'Spam', 'Sent', 'Check My Email']
        icons = ['📥', '⚠️', '📤', '🔍']
        
        for i, (folder, icon) in enumerate(zip(folders, icons)):
            if st.sidebar.button(f"{icon} {folder}", key=f'sidebar_{folder}', 
                               use_container_width=True):
                st.session_state.folder = folder
                st.session_state.selected = (folder.lower().replace(' ', '_'), None)
        
        st.markdown("---")
        
        # Stats in sidebar
        st.subheader("Statistics")
        st.markdown("""
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-value">{}</div>
                <div class="stat-label">Inbox</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{}</div>
                <div class="stat-label">Spam</div>
            </div>
        </div>
        """.format(len(st.session_state.inbox), len(st.session_state.spam)), unsafe_allow_html=True)
        
        # Simulate incoming emails button
        if st.button("📩 Receive new email", use_container_width=True):
            if st.session_state.pointer < len(PREDEFINED_EMAILS):
                email = PREDEFINED_EMAILS[st.session_state.pointer]
                label = pred(email['body'], w2v_model, model)
                entry = {
                    'subject': email['subject'], 
                    'body': email['body'], 
                    'pred': label,
                    'date': email['date']
                }
                (st.session_state.spam if label else st.session_state.inbox).append(entry)
                st.session_state.pointer += 1
    
    # Main content area
    # Simulate incoming emails on auto every time app renders
    if st.session_state.pointer < len(PREDEFINED_EMAILS):
        email = PREDEFINED_EMAILS[st.session_state.pointer]
        label = pred(email['body'], w2v_model, model)
        entry = {
            'subject': email['subject'], 
            'body': email['body'], 
            'pred': label,
            'date': email['date']
        }
        (st.session_state.spam if label else st.session_state.inbox).append(entry)
        st.session_state.pointer += 1

    # Display folder header
    st.title(f"{st.session_state.folder}")
    
    # Display based on folder
    fld = st.session_state.folder
    if fld in ['Inbox', 'Spam', 'Sent']:
        folder_key = fld.lower()
        msgs = getattr(st.session_state, folder_key)
        
        if not msgs:
            st.info(f"Your {fld} is empty.")
        else:
            # Two-column layout for email list and preview
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Messages")
                for idx, msg in enumerate(msgs):
                    # Check if email has been viewed
                    is_read = (folder_key, idx) in st.session_state.viewed_emails
                    
                    # Create interactive email list item
                    email_class = "email-list selected" if st.session_state.selected == (folder_key, idx) else "email-list"
                    if not is_read:
                        email_class += " unread"
                        
                    html = f"""
                    <div class="{email_class}" id="{folder_key}_{idx}" onclick="this.style.backgroundColor='#2D1F3D';">
                        <strong>{msg['subject']}</strong><br>
                        <small style="color: #aaa;">{format_date(msg['date'])}</small><br>
                        <span style="color: #999; font-size: 0.9em;">{get_email_preview(msg['body'])}</span>
                    </div>
                    """
                    
                    # Use button to make it clickable
                    if st.markdown(html, unsafe_allow_html=True):
                        pass
                    
                    # Separate button to actually handle the click (Streamlit limitation)
                    if st.button(f"View", key=f"{folder_key}_btn_{idx}", use_container_width=True):
                        st.session_state.selected = (folder_key, idx)
                        st.session_state.viewed_emails.add((folder_key, idx))
            
            with col2:
                # Email detail view
                f, id = st.session_state.selected
                if f == folder_key and id is not None and id < len(msgs):
                    m = msgs[id]
                    
                    # Add to viewed emails
                    st.session_state.viewed_emails.add((folder_key, id))
                    
                    st.markdown(f"""
                    <div class="email-detail">
                        <div class="email-header">
                            <h3>{m['subject']}</h3>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="color: #aaa;">{format_date(m['date'])}</div>
                                <div>
                                    <span class="{'spam-badge' if m['pred'] else 'safe-badge'}">
                                        {"⚠️ SPAM" if m['pred'] else "✅ SAFE"}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div class="email-body">{m['body'].replace('\n', '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("🗑️ Delete", use_container_width=True):
                            msgs.pop(id)
                            st.session_state.selected = (folder_key, None)
                    with col_b:
                        if not m['pred'] and folder_key == "inbox":
                            if st.button("⚠️ Mark as Spam", use_container_width=True):
                                m['pred'] = 1
                                st.session_state.spam.append(m)
                                msgs.pop(id)
                                st.session_state.selected = ("spam", len(st.session_state.spam) - 1)
                        elif m['pred'] and folder_key == "spam":
                            if st.button("✅ Not Spam", use_container_width=True):
                                m['pred'] = 0
                                st.session_state.inbox.append(m)
                                msgs.pop(id)
                                st.session_state.selected = ("inbox", len(st.session_state.inbox) - 1)
                    with col_c:
                        if st.button("↩️ Reply", use_container_width=True):
                            st.session_state.folder = 'Check My Email'
                            st.session_state.selected = ('check_my_email', None)
                else:
                    st.info("Select an email to view its contents")
    
    elif fld == 'Check My Email':
        st.subheader("Check if a message is spam")
        
        # Input area with improved styling
        st.markdown('<p style="color: #aaa; margin-bottom: 0.5rem;">Subject:</p>', unsafe_allow_html=True)
        subject = st.text_input("", key="check_subject", placeholder="Enter email subject...")
        
        st.markdown('<p style="color: #aaa; margin-bottom: 0.5rem;">Message:</p>', unsafe_allow_html=True)
        txt = st.text_area("", key="user_input", placeholder="Enter email content to check if it's spam...", height=200)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Check Now", key="check"):
                if not txt:
                    st.error("Please enter some text to check")
                else:
                    res = pred(txt, w2v_model, model)
                    result_text = "🚨 SPAM DETECTED" if res else "✅ SAFE MESSAGE"
                    result_color = "#f44336" if res else "#4CAF50"
                    
                    st.markdown(f"""
                    <div style="background-color: {result_color}20; border-left: 4px solid {result_color}; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                        <h3 style="color: {result_color};">{result_text}</h3>
                        <p>Our AI has analyzed your message and determined it is {' spam.' if res else ' not spam.'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to sent messages
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.sent.append({
                        'subject': subject if subject else 'No Subject',
                        'body': txt,
                        'pred': res,
                        'date': current_time
                    })
        
        # Display confidence explanation
        st.markdown("""
        <div class="feature-card" style="margin-top: 2rem;">
            <h4>How our detection works</h4>
            <p>Our spam detection system uses a machine learning model that analyzes the content and structure of messages to determine if they're spam. The model was trained on a diverse dataset from three sources and uses Word2Vec to analyze semantic relationships between words.</p>
            <p>Common spam indicators include:</p>
            <ul>
                <li>Urgent calls to action</li>
                <li>Requests for personal information</li>
                <li>Suspicious links</li>
                <li>Exaggerated promises or offers</li>
                <li>Poor grammar and spelling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---- Render Page ----
if st.session_state.page == 'home':
    render_home()
else:
    render_app()

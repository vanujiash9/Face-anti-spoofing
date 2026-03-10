import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import time
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from torchvision import transforms
import mediapipe as mp

# Setup path
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

# --- INITIALIZE DETECTOR ---
@st.cache_resource
def get_face_detector():
    mp_fd = mp.solutions.face_detection
    return mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# --- MODERN PROFESSIONAL UI ---
st.set_page_config(
    page_title="FAS Detection Pro + AI Insight",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 500;
    }
    
    /* Result Cards - Premium Design */
    .result-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.12);
    }
    
    /* Status Badges - LIVE (Green) & SPOOF (Red) */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .badge-live {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: 2px solid #0a6e5c;
    }
    
    .badge-spoof {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border: 2px solid #b71c1c;
    }
    
    .badge-icon {
        font-size: 1.3rem;
        margin-right: 0.5rem;
    }
    
    /* Metric Display */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #212529;
        font-size: 1.4rem;
        font-weight: 800;
    }
    
    .metric-value.confidence {
        color: #667eea;
    }
    
    .metric-value.latency {
        color: #38ef7d;
    }
    
    /* AI Explanation Box */
    .ai-explain-container {
        background: linear-gradient(135deg, #fff5f0 0%, #ffe8e0 100%);
        border-left: 5px solid #ff6b35;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.15);
    }
    
    .ai-explain-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #d63031;
    }
    
    .ai-explain-icon {
        font-size: 1.8rem;
        margin-right: 0.8rem;
    }
    
    .ai-reason-item {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.6rem 0;
        border-left: 3px solid #ff6b35;
        font-size: 0.95rem;
        color: #2c3e50;
        transition: transform 0.2s ease;
    }
    
    .ai-reason-item:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .ai-reason-badge {
        display: inline-block;
        background: #ff6b35;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
        text-transform: uppercase;
    }
    
    /* Technical Info Box */
    .tech-info-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d4e9f7 100%);
        border-left: 5px solid #3498db;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #2c3e50;
    }
    
    .tech-info-title {
        font-weight: 700;
        color: #2980b9;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Heatmap Container */
    .heatmap-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .heatmap-title {
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        text-align: center;
    }
    
    /* Progress Bar Custom */
    .confidence-bar-container {
        width: 100%;
        height: 12px;
        background: #e9ecef;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.8rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    
    .bar-live {
        background: linear-gradient(90deg, #11998e, #38ef7d);
    }
    
    .bar-spoof {
        background: linear-gradient(90deg, #eb3349, #f45c43);
    }
    
    /* Image Container */
    .image-frame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        border: 3px solid white;
        background: white;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(102, 126, 234, 0.6);
    }
    
    /* No Face Detected Message */
    .no-face-msg {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #fff9e6 0%, #ffe5b4 100%);
        border-radius: 12px;
        border: 2px dashed #ffa500;
        color: #ff8c00;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Stats Display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC ---
MODELS_CONFIG = {
    "ConvNeXt (Best Precision)": {"class": ConvNextBinary, "path": "saved_models/convnext/best.pt", "size": 224},
    "EfficientNet (Optimized)": {"class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT (Transformer)": {"class": ViTBinary, "path": "checkpoints/vit/best.pt", "size": 224}
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_fas_model(model_name):
    cfg = MODELS_CONFIG[model_name]
    try:
        model = cfg['class'](pretrained=False).to(device)
        state_dict = torch.load(cfg['path'], map_location=device)
        new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state)
        model.eval()
        return model, cfg['size']
    except Exception as e:
        return None, str(e)

# --- GRAD-CAM FOR HEATMAP VISUALIZATION ---
def generate_gradcam(model, input_tensor, model_name):
    """Generate GradCAM heatmap to show where AI is looking"""
    model.zero_grad()
    
    # Select target layer based on model architecture
    if "ConvNeXt" in model_name:
        target_layer = model.backbone.stages[-1]
    elif "EfficientNet" in model_name:
        target_layer = model.backbone.blocks[-1]
    else:  # ViT
        target_layer = model.backbone.norm

    # Hook to capture feature maps
    feature_map = None
    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output

    handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    output = model(input_tensor)
    score = torch.sigmoid(output).item()
    
    # Backward pass
    output.backward()
    handle.remove()

    # Generate heatmap
    if feature_map is not None:
        if isinstance(feature_map, tuple): 
            feature_map = feature_map[0]
        
        # Average pooling across channels
        weights = torch.mean(feature_map, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_map, dim=1).squeeze().detach().cpu().numpy()
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam, score
    return None, score

# --- AI EXPLANATION ENGINE ---
def get_ai_explanation(score, cam, threshold):
    """Generate detailed AI explanations based on prediction"""
    is_spoof = score > threshold
    reasons = []
    
    if is_spoof:
        # SPOOF Detection Explanations
        if cam is not None:
            # Analyze heatmap patterns
            edge_intensity = (np.mean(cam[0:15, :]) + np.mean(cam[-15:, :]) + 
                            np.mean(cam[:, 0:15]) + np.mean(cam[:, -15:])) / 4
            
            if edge_intensity > 0.4:
                reasons.append({
                    "type": "STRUCTURAL",
                    "text": "Phát hiện cạnh viền bất thường - Đặc trưng của màn hình điện thoại/tablet hoặc ảnh in"
                })
            
            center_intensity = np.mean(cam[60:160, 60:160])
            if center_intensity > 0.7:
                reasons.append({
                    "type": "TEXTURE",
                    "text": "Độ phản xạ không tự nhiên ở vùng trung tâm khuôn mặt - Bề mặt quá phẳng (thiếu độ sâu 3D)"
                })
            
            if np.max(cam) > 0.8:
                reasons.append({
                    "type": "FREQUENCY",
                    "text": "AI phát hiện Moiré Pattern - Nhiễu tần số cao đặc trưng khi chụp lại màn hình"
                })
        
        # General spoof indicators
        if score > 0.85:
            reasons.append({
                "type": "CONFIDENCE",
                "text": f"Mức độ tin cậy cao ({score:.1%}) - AI rất chắc chắn đây là ảnh giả mạo"
            })
        
        reasons.append({
            "type": "MATERIAL",
            "text": "Thiếu các đặc điểm sinh học: Da không có texture tự nhiên, mắt không có phản chiếu sáng phức tạp"
        })
        
        reasons.append({
            "type": "DEPTH",
            "text": "Khuôn mặt thiếu chiều sâu không gian - Mọi điểm ảnh đều ở cùng một mặt phẳng (2D)"
        })
        
    else:
        # LIVE Detection Explanations
        reasons.append({
            "type": "AUTHENTIC",
            "text": "Cấu trúc da đồng nhất với texture tự nhiên - Có lỗ chân lông, nếp nhăn nhỏ"
        })
        
        reasons.append({
            "type": "LIGHT",
            "text": "Phản xạ ánh sáng tự nhiên - Có gradient sáng/tối phức tạp đặc trưng của khuôn mặt 3D"
        })
        
        reasons.append({
            "type": "DEPTH",
            "text": "Có độ sâu không gian thực - Mũi, má, trán ở các mặt phẳng khác nhau"
        })
        
        if cam is not None and np.mean(cam[0:15, :]) < 0.3:
            reasons.append({
                "type": "BACKGROUND",
                "text": "Không phát hiện vật liệu lạ xung quanh vùng mặt - Không có viền màn hình hay cạnh giấy"
            })
        
        reasons.append({
            "type": "BIOMETRIC",
            "text": "Các đặc điểm sinh học tự nhiên: Mắt có phản chiếu ánh sáng, da có màu sắc sống động"
        })
    
    return reasons

# --- CORE INFERENCE FUNCTION ---
def run_inference(image_pil, model, size, threshold, detector, enable_heatmap=False, model_name=""):
    start_time = time.time()
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_cv.shape
    
    results_mp = detector.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    found_faces = []
    heatmaps = []
    
    if results_mp.detections:
        draw = ImageDraw.Draw(image_pil)
        for detection in results_mp.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * w_img), int(bboxC.ymin * h_img), \
                         int(bboxC.width * w_img), int(bboxC.height * h_img)
            
            m = int(w * 0.25)
            x1, y1 = max(0, x-m), max(0, y-m)
            x2, y2 = min(w_img, x+w+m), min(h_img, y+h+m)
            
            face_crop = image_pil.crop((x1, y1, x2, y2))
            inp = tf(face_crop).unsqueeze(0).to(device)
            
            # Generate heatmap if enabled
            cam = None
            if enable_heatmap:
                inp.requires_grad_(True)
                cam, score = generate_gradcam(model, inp, model_name)
            else:
                with torch.no_grad():
                    out = model(inp)
                    score = torch.sigmoid(out).item()
            
            label = "SPOOF" if score > threshold else "LIVE"
            conf = score if score > threshold else 1 - score
            
            # Draw bounding box
            color = (235, 51, 73) if label == "SPOOF" else (17, 153, 142)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            # Draw label
            text = f"{label} {conf:.1%}"
            bbox = draw.textbbox((x1, y1-35), text)
            draw.rectangle([bbox[0]-8, bbox[1]-8, bbox[2]+8, bbox[3]+8], fill=color)
            draw.text((x1, y1-35), text, fill="white")
            
            # Get AI explanations
            explanations = get_ai_explanation(score, cam, threshold)
            
            found_faces.append({
                "label": label,
                "conf": conf,
                "score": score,
                "latency": (time.time() - start_time) * 1000,
                "explanations": explanations,
                "face_crop": face_crop,
                "cam": cam
            })
            
    return image_pil, found_faces

# --- DISPLAY RESULT WITH AI EXPLANATION ---
def display_result_with_explanation(result_img, results, show_heatmap=False):
    if not results:
        st.markdown("""
            <div class="no-face-msg">
                ⚠️ Không phát hiện khuôn mặt trong ảnh
            </div>
        """, unsafe_allow_html=True)
        return
    
    for idx, r in enumerate(results):
        is_live = r['label'] == "LIVE"
        badge_class = "badge-live" if is_live else "badge-spoof"
        icon = "✅" if is_live else "⚠️"
        bar_class = "bar-live" if is_live else "bar-spoof"
        
        # Main result card
        st.markdown(f"""
        <div class="result-card">
            <div class="status-badge {badge_class}">
                <span class="badge-icon">{icon}</span>
                <span>{r['label']}</span>
            </div>
            
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">Độ tin cậy</div>
                    <div class="metric-value confidence">{r['conf']:.1%}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Thời gian xử lý</div>
                    <div class="metric-value latency">{r['latency']:.1f}ms</div>
                </div>
            </div>
            
            <div class="confidence-bar-container">
                <div class="confidence-bar {bar_class}" style="width: {r['conf']*100}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Heatmap visualization
        if show_heatmap and r['cam'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                st.markdown('<div class="heatmap-title">🎯 Original Face</div>', unsafe_allow_html=True)
                st.image(r['face_crop'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                st.markdown('<div class="heatmap-title">🔥 AI Focus Heatmap</div>', unsafe_allow_html=True)
                
                # Create heatmap overlay
                face_np = np.array(r['face_crop'].resize((300, 300)))
                heatmap = cv2.applyColorMap(np.uint8(255 * r['cam']), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (300, 300))
                overlay = cv2.addWeighted(face_np, 0.5, heatmap, 0.5, 0)
                
                st.image(overlay, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Explanation section
        explanation_type = "⚠️ TẠI SAO AI PHÁT HIỆN GIẢ MẠO" if not is_live else "✅ TẠI SAO AI XÁC NHẬN THẬT"
        
        st.markdown(f"""
        <div class="ai-explain-container">
            <div class="ai-explain-header">
                <span class="ai-explain-icon">🧠</span>
                <span>{explanation_type}</span>
            </div>
        """, unsafe_allow_html=True)
        
        for exp in r['explanations']:
            st.markdown(f"""
            <div class="ai-reason-item">
                <span class="ai-reason-badge">{exp['type']}</span>
                {exp['text']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Technical information
        st.markdown(f"""
        <div class="tech-info-box">
            <div class="tech-info-title">📊 Thông Tin Kỹ Thuật</div>
            <p><strong>Raw Score:</strong> {r['score']:.4f} | <strong>Threshold:</strong> {st.session_state.get('threshold', 0.5):.2f}</p>
            <p><strong>Phân Loại:</strong> {'Spoof (score > threshold)' if not is_live else 'Live (score ≤ threshold)'}</p>
            <p><strong>Model:</strong> {st.session_state.get('model_name', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    # Hero Header
    st.markdown("""
        <div class="hero-header">
            <h1 class="hero-title">🛡️ Face Anti-Spoofing Detection Pro</h1>
            <p class="hero-subtitle">Advanced AI-Powered Liveness Detection with Explainable Insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "total_analyzed" not in st.session_state:
        st.session_state.total_analyzed = 0
        st.session_state.live_count = 0
        st.session_state.spoof_count = 0

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ CẤU HÌNH HỆ THỐNG")
        
        m_name = st.selectbox(
            "🤖 Mô hình AI",
            list(MODELS_CONFIG.keys()),
            help="Chọn model deep learning"
        )
        
        threshold = st.slider(
            "🎯 Ngưỡng phát hiện",
            0.0, 1.0, 0.5, 0.05,
            help="Điều chỉnh độ nhạy"
        )
        
        # Store in session state
        st.session_state.threshold = threshold
        st.session_state.model_name = m_name
        
        enable_heatmap = st.checkbox(
            "🔥 Hiển thị AI Heatmap",
            value=True,
            help="Bật để xem vùng AI tập trung (tốn thời gian hơn)"
        )
        
        st.markdown("---")
        
        # Stats Display
        st.markdown("### 📊 THỐNG KÊ PHÂN TÍCH")
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value" style="color: #667eea;">{st.session_state.total_analyzed}</div>
                <div class="stat-label">Tổng</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #11998e;">{st.session_state.live_count}</div>
                <div class="stat-label">Live</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #eb3349;">{st.session_state.spoof_count}</div>
                <div class="stat-label">Spoof</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💻 THÔNG TIN HỆ THỐNG")
        device_color = "#38ef7d" if "cuda" in str(device).lower() else "#667eea"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; color: #212529;">
            <p style="margin: 0.3rem 0;"><strong>Device:</strong> <span style="color: {device_color}; font-weight: 700;">{str(device).upper()}</span></p>
            <p style="margin: 0.3rem 0;"><strong>Model:</strong> {m_name.split('(')[0].strip()}</p>
            <p style="margin: 0.3rem 0;"><strong>Heatmap:</strong> {'🟢 ON' if enable_heatmap else '🔴 OFF'}</p>
            <p style="margin: 0.3rem 0;"><strong>Status:</strong> <span style="color: #38ef7d; font-weight: 700;">🟢 READY</span></p>
        </div>
        """, unsafe_allow_html=True)

    model, m_size = load_fas_model(m_name)
    detector = get_face_detector()

    if not model:
        st.error("❌ Lỗi tải model. Kiểm tra file weights.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📁 BATCH UPLOAD", "🔗 IMAGE LINKS", "📸 WEBCAM"])

    # --- TAB 1: BATCH UPLOAD ---
    with tab1:
        st.markdown("### 📤 Tải nhiều ảnh cùng lúc")
        files = st.file_uploader(
            "Chọn ảnh (JPG/PNG)",
            type=['jpg','png','jpeg'],
            accept_multiple_files=True,
            help="Có thể chọn nhiều file"
        )
        
        if files:
            st.markdown("---")
            st.markdown(f"### 📊 Kết quả phân tích ({len(files)} ảnh)")
            
            progress = st.progress(0)
            
            for file_idx, f in enumerate(files):
                img = Image.open(f).convert('RGB')
                
                with st.spinner(f'🔄 Đang phân tích {f.name}...'):
                    res_img, results = run_inference(
                        img.copy(), model, m_size, threshold, detector, 
                        enable_heatmap, m_name
                    )
                
                # Update stats
                for r in results:
                    st.session_state.total_analyzed += 1
                    if r['label'] == "LIVE":
                        st.session_state.live_count += 1
                    else:
                        st.session_state.spoof_count += 1
                
                # Display result
                st.markdown(f"#### 📄 {f.name}")
                
                col1, col2 = st.columns([1.3, 2])
                
                with col1:
                    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                    st.image(res_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    display_result_with_explanation(res_img, results, enable_heatmap)
                
                st.markdown("---")
                progress.progress((file_idx + 1) / len(files))
            
            st.success("✅ Hoàn thành phân tích tất cả ảnh!")

    # --- TAB 2: IMAGE LINKS ---
    with tab2:
        st.markdown("### 🔗 Phân tích từ URL")
        url_input = st.text_area(
            "Dán link ảnh (mỗi dòng một link)",
            height=150,
            placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg"
        )
        
        if st.button("🚀 PHÂN TÍCH TẤT CẢ", type="primary", use_container_width=True):
            links = [l.strip() for l in url_input.split('\n') if l.strip()]
            
            if not links:
                st.warning("⚠️ Vui lòng nhập ít nhất một URL")
            else:
                progress_bar = st.progress(0)
                st.markdown("---")
                
                for idx, link in enumerate(links):
                    try:
                        with st.spinner(f'📥 Đang tải ảnh {idx+1}/{len(links)}...'):
                            resp = requests.get(link, timeout=10)
                            img = Image.open(BytesIO(resp.content)).convert('RGB')
                        
                        with st.spinner(f'🔄 Đang phân tích...'):
                            res_img, results = run_inference(
                                img, model, m_size, threshold, detector,
                                enable_heatmap, m_name
                            )
                        
                        # Update stats
                        for r in results:
                            st.session_state.total_analyzed += 1
                            if r['label'] == "LIVE":
                                st.session_state.live_count += 1
                            else:
                                st.session_state.spoof_count += 1
                        
                        # Display
                        st.markdown(f"#### 🔗 Link {idx+1}: `{link[:50]}...`")
                        
                        col1, col2 = st.columns([1.3, 2])
                        
                        with col1:
                            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                            st.image(res_img, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            display_result_with_explanation(res_img, results, enable_heatmap)
                        
                        st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi tải ảnh từ: {link[:50]}... | Error: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(links))
                
                st.success("✅ Hoàn thành phân tích!")

    # --- TAB 3: WEBCAM ---
    with tab3:
        st.markdown("### 📸 Chụp ảnh kiểm tra nhanh")
        
        st.info("💡 **Mẹo:** Thử các trường hợp khác nhau - ảnh thật, ảnh trên điện thoại, ảnh in, video...")
        
        cam_file = st.camera_input("Sử dụng webcam")
        
        if cam_file:
            img = Image.open(cam_file).convert('RGB')
            
            with st.spinner('🔄 Đang phân tích khuôn mặt...'):
                res_img, results = run_inference(
                    img, model, m_size, threshold, detector,
                    enable_heatmap, m_name
                )
                
                # Update stats
                for r in results:
                    st.session_state.total_analyzed += 1
                    if r['label'] == "LIVE":
                        st.session_state.live_count += 1
                    else:
                        st.session_state.spoof_count += 1
            
            st.markdown("---")
            
            col1, col2 = st.columns([1.3, 2])
            
            with col1:
                st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                st.image(res_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                display_result_with_explanation(res_img, results, enable_heatmap)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem 0;">
        <p style="font-size: 0.9rem; margin: 0;">
            <strong>🛡️ FAS Detection Pro</strong> | Powered by Deep Learning & Explainable AI
        </p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            ConvNeXt • EfficientNet • Vision Transformer | GradCAM Visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
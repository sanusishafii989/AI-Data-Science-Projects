# ================================================================
# CropGuard AI -- Streamlit App
# Offline-first Multi-Crop Disease Identification System
# NextGen Innovation Challenge 2026
# ================================================================

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# Page configuration -- must be the very first Streamlit call
# ----------------------------------------------------------------
st.set_page_config(
    page_title='CropGuard AI',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ================================================================
# CONSTANTS -- class names and treatment advice
# ================================================================

DISPLAY_NAMES = [
    'Tomato - Early Blight',
    'Tomato - Late Blight',
    'Tomato - Leaf Mold',
    'Tomato - Healthy',
    'Bell Pepper - Bacterial Spot',
    'Bell Pepper - Healthy',
    'Maize - Common Rust',
    'Maize - Northern Leaf Blight',
    'Maize - Healthy',
    'Potato - Early Blight',
    'Potato - Late Blight',
    'Potato - Healthy',
    'Vine Crop - Black Rot',
    'Vine Crop - Black Measles',
    'Vine Crop - Healthy',
]

IMAGE_SIZE = 224

# ----------------------------------------------------------------
# Treatment advice dictionary -- one entry per class
# Each entry: symptoms, cause, treatment steps, urgency
# ----------------------------------------------------------------
TREATMENT_ADVICE = {
    'Tomato - Early Blight': {
        'emoji': '🍅',
        'crop': 'Tomato',
        'condition': 'Early Blight',
        'symptoms': 'Dark brown spots with concentric yellow rings on older lower leaves.',
        'cause': 'Fungal infection (Alternaria solani)',
        'urgency': '🟡 Moderate',
        'treatment': [
            'Remove and destroy all infected leaves immediately',
            'Apply copper-based fungicide (copper oxychloride 50WP) — spray every 7 days',
            'Switch to drip irrigation; avoid wetting the leaves',
            'Apply organic mulch around the base to prevent soil splash',
            'Do NOT plant tomato in the same plot for at least 2 seasons',
        ],
        'prevention': 'Use certified disease-free seeds. Space plants 60cm apart for air flow.',
        'cost_estimate': '₦800–₦1,500 per application (copper fungicide)',
    },
    'Tomato - Late Blight': {
        'emoji': '🍅',
        'crop': 'Tomato',
        'condition': 'Late Blight',
        'symptoms': 'Water-soaked dark lesions on leaves and stems; white mold under leaves in humid conditions.',
        'cause': 'Oomycete pathogen (Phytophthora infestans) — spreads very fast',
        'urgency': '🔴 URGENT — Act within 24 hours',
        'treatment': [
            'IMMEDIATELY remove and bag all visibly infected plants — do NOT compost',
            'Apply systemic fungicide containing metalaxyl + mancozeb',
            'Spray the entire field, including soil around plants',
            'Stop all overhead irrigation at once',
            'If more than 40% of plants are affected, consider destroying the crop to save neighbouring farms',
        ],
        'prevention': 'Plant resistant varieties (e.g. Tylka F1). Avoid planting in cool, humid conditions.',
        'cost_estimate': '₦1,200–₦2,500 per application (metalaxyl-based)',
    },
    'Tomato - Leaf Mold': {
        'emoji': '🍅',
        'crop': 'Tomato',
        'condition': 'Leaf Mold',
        'symptoms': 'Pale yellow spots on upper leaf surface; olive-green to brown mold patches underneath.',
        'cause': 'Fungal infection (Passalora fulva) — mainly in high-humidity environments',
        'urgency': '🟡 Moderate',
        'treatment': [
            'Improve greenhouse or canopy ventilation immediately',
            'Reduce irrigation frequency and avoid evening watering',
            'Remove heavily infected leaves and dispose properly',
            'Apply copper-based or chlorothalonil fungicide every 10 days',
        ],
        'prevention': 'Maintain plant spacing. Use resistant varieties. Keep humidity below 85%.',
        'cost_estimate': '₦600–₦1,200 per application',
    },
    'Tomato - Healthy': {
        'emoji': '✅',
        'crop': 'Tomato',
        'condition': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'cause': 'N/A',
        'urgency': '🟢 No action required',
        'treatment': [
            'Continue your current farming practices — they are working well',
            'Monitor plants weekly for early signs of disease',
            'Apply preventive copper spray during rainy season as a precaution',
            'Ensure consistent watering and balanced NPK fertilisation',
        ],
        'prevention': 'Inspect leaves regularly. Remove weeds that may harbour pests.',
        'cost_estimate': 'Minimal — preventive spray ₦300–₦500 per application',
    },
    'Bell Pepper - Bacterial Spot': {
        'emoji': '🫑',
        'crop': 'Bell Pepper',
        'condition': 'Bacterial Spot',
        'symptoms': 'Small water-soaked spots turning brown with yellow halos on leaves; fruit may show raised scabs.',
        'cause': 'Bacterial infection (Xanthomonas campestris)',
        'urgency': '🟡 Moderate',
        'treatment': [
            'Spray copper-based bactericide (copper hydroxide or copper oxychloride)',
            'Remove and destroy badly infected leaves',
            'Avoid overhead watering — use drip irrigation',
            'Do NOT work among plants when they are wet',
            'Rotate crops for at least 2 years',
        ],
        'prevention': 'Use disease-free certified seedlings. Sterilise tools between uses.',
        'cost_estimate': '₦700–₦1,400 per application (copper bactericide)',
    },
    'Bell Pepper - Healthy': {
        'emoji': '✅',
        'crop': 'Bell Pepper',
        'condition': 'Healthy',
        'symptoms': 'No disease detected.',
        'cause': 'N/A',
        'urgency': '🟢 No action required',
        'treatment': [
            'Continue current practices',
            'Inspect every 5–7 days during rainy season',
            'Apply balanced fertiliser as per soil test recommendation',
        ],
        'prevention': 'Keep the area weed-free and ensure good drainage.',
        'cost_estimate': 'Minimal',
    },
    'Maize - Common Rust': {
        'emoji': '🌽',
        'crop': 'Maize',
        'condition': 'Common Rust',
        'symptoms': 'Brick-red to brown pustules (raised powdery spots) on both leaf surfaces.',
        'cause': 'Fungal infection (Puccinia sorghi) — spreads by wind',
        'urgency': '🟡 Moderate',
        'treatment': [
            'Apply azoxystrobin or mancozeb-based fungicide at early appearance',
            'Scout every 7 days; spray again if pustules continue to spread',
            'Harvest early if infection is severe and the cobs are ready',
            'For next season: plant rust-resistant varieties (e.g., SAMMAZ series)',
        ],
        'prevention': 'Early planting (before heavy rains) reduces rust incidence significantly.',
        'cost_estimate': '₦900–₦1,800 per application',
    },
    'Maize - Northern Leaf Blight': {
        'emoji': '🌽',
        'crop': 'Maize',
        'condition': 'Northern Leaf Blight',
        'symptoms': 'Long, elliptical grey-green to tan lesions (cigar-shaped) running along leaves.',
        'cause': 'Fungal infection (Exserohilum turcicum)',
        'urgency': '🟠 High',
        'treatment': [
            'Apply foliar fungicide (propiconazole or azoxystrobin) immediately',
            'Remove and burn heavily infected crop residue after harvest',
            'Do NOT leave infected stalks in the field over the off-season',
            'For next planting: use certified resistant hybrid seed',
        ],
        'prevention': 'Crop rotation with legumes. Deep tillage to bury infected residue.',
        'cost_estimate': '₦1,000–₦2,000 per application',
    },
    'Maize - Healthy': {
        'emoji': '✅',
        'crop': 'Maize',
        'condition': 'Healthy',
        'symptoms': 'No disease detected.',
        'cause': 'N/A',
        'urgency': '🟢 No action required',
        'treatment': [
            'Continue your current practices',
            'Apply top-dressing nitrogen fertiliser at knee-height stage',
            'Monitor weekly, especially around tasselling stage',
        ],
        'prevention': 'Ensure adequate plant spacing (75cm between rows) for airflow.',
        'cost_estimate': 'Minimal',
    },
    'Potato - Early Blight': {
        'emoji': '🥔',
        'crop': 'Potato',
        'condition': 'Early Blight',
        'symptoms': 'Dark brown bull\'s-eye spots with yellow rings, appearing on older leaves first.',
        'cause': 'Fungal infection (Alternaria solani)',
        'urgency': '🟡 Moderate',
        'treatment': [
            'Remove all infected lower leaves and destroy',
            'Apply chlorothalonil or mancozeb fungicide every 7–10 days',
            'Avoid working among plants when leaves are wet',
            'Ensure adequate potassium fertilisation — stressed plants are more susceptible',
        ],
        'prevention': 'Plant certified seed tubers. Rotate crops for 3 years minimum.',
        'cost_estimate': '₦600–₦1,200 per application',
    },
    'Potato - Late Blight': {
        'emoji': '🥔',
        'crop': 'Potato',
        'condition': 'Late Blight',
        'symptoms': 'Dark water-soaked lesions on leaves, rapidly spreading; infected tubers show dark rot.',
        'cause': 'Oomycete pathogen (Phytophthora infestans) — can destroy an entire field in days',
        'urgency': '🔴 URGENT — Act immediately',
        'treatment': [
            'STOP all irrigation immediately',
            'Apply metalaxyl + mancozeb (Ridomil Gold MZ) systemic fungicide NOW',
            'Destroy all visible infected plants — bury or burn, do NOT compost',
            'If tubers are mature, harvest immediately before the rot spreads underground',
            'Notify neighbouring farmers — this disease spreads on wind and rain',
        ],
        'prevention': 'Use certified seed. Apply preventive fungicide before rains. Plant in well-drained soil.',
        'cost_estimate': '₦1,500–₦3,000 per application (emergency treatment)',
    },
    'Potato - Healthy': {
        'emoji': '✅',
        'crop': 'Potato',
        'condition': 'Healthy',
        'symptoms': 'No disease detected.',
        'cause': 'N/A',
        'urgency': '🟢 No action required',
        'treatment': [
            'Continue current management practices',
            'Apply preventive fungicide before the rainy season as insurance',
            'Check for aphids (late blight vectors) weekly',
        ],
        'prevention': 'Earth up plants to protect tubers. Remove volunteer potatoes from previous seasons.',
        'cost_estimate': 'Minimal',
    },
    'Vine Crop - Black Rot': {
        'emoji': '🍇',
        'crop': 'Vine / Fruit Crop',
        'condition': 'Black Rot',
        'symptoms': 'Brown circular spots with dark borders on leaves; infected fruit turns black and shrivels.',
        'cause': 'Fungal infection (Guignardia bidwellii)',
        'urgency': '🟠 High',
        'treatment': [
            'Remove and destroy all mummified (shrivelled) fruit immediately',
            'Apply myclobutanil or captan fungicide at 10-day intervals',
            'Prune to improve air circulation through the canopy',
            'Clear all leaf litter and fallen fruit from around plants',
        ],
        'prevention': 'Annual pruning for open canopy. Apply dormant-season copper spray.',
        'cost_estimate': '₦800–₦1,600 per application',
    },
    'Vine Crop - Black Measles': {
        'emoji': '🍇',
        'crop': 'Vine / Fruit Crop',
        'condition': 'Black Measles (Esca)',
        'symptoms': 'Interveinal yellowing or reddening; fruit shows dark purple spots; internal wood turns brown.',
        'cause': 'Complex fungal infection of the wood (Phaeomoniella, Phaeoacremonium)',
        'urgency': '🟠 High — no complete cure',
        'treatment': [
            'No chemical cure currently exists — focus on management',
            'Remove and burn all dead wood and badly affected canes',
            'Paint pruning wounds immediately with wound sealant or copper paste',
            'Avoid pruning during wet weather — spores spread in rain',
            'Apply sodium arsenite (where legally permitted) as trunk injection — consult agronomist',
        ],
        'prevention': 'Sterilise pruning tools between every plant. Avoid large pruning wounds.',
        'cost_estimate': 'Labour-intensive — wound treatment ₦500–₦1,000 per plant',
    },
    'Vine Crop - Healthy': {
        'emoji': '✅',
        'crop': 'Vine / Fruit Crop',
        'condition': 'Healthy',
        'symptoms': 'No disease detected.',
        'cause': 'N/A',
        'urgency': '🟢 No action required',
        'treatment': [
            'Continue current management',
            'Annual dormant-season copper spray as preventive measure',
            'Maintain canopy pruning for good air flow',
        ],
        'prevention': 'Keep tools sterilised. Remove weeds competing with the crop.',
        'cost_estimate': 'Minimal',
    },
}

# ================================================================
# MODEL LOADING -- cached so it only loads once per session
# ================================================================

@st.cache_resource(show_spinner='Loading AI model...')
def load_model():
    """
    Loads the Keras model from disk. Returns None if file not found
    so the app can show a helpful error instead of crashing.
    """
    model_path = 'crop_disease_model.h5'
    if not os.path.exists(model_path):
        return None
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(model_path)
        return m
    except Exception as exc:
        st.error(f"Model load error: {exc}")
        return None


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resizes a PIL image to IMAGE_SIZE x IMAGE_SIZE, normalises to
    [0, 1] float32, and adds the batch dimension.

    Args:
        pil_image : PIL.Image in RGB mode.

    Returns:
        numpy array of shape (1, IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    img = pil_image.convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_prediction(model, pil_image: Image.Image):
    """
    Runs model inference on a single image.

    Returns:
        predicted_class : str   -- display name of top prediction
        confidence      : float -- probability of top prediction (0-1)
        all_probs       : list  -- probability for each of the 15 classes
    """
    inp         = preprocess_image(pil_image)
    probs       = model.predict(inp, verbose=0)[0]
    top_idx     = int(np.argmax(probs))
    return DISPLAY_NAMES[top_idx], float(probs[top_idx]), probs

# ================================================================
# GPS LOGGING -- save predictions to a CSV file
# ================================================================

LOG_FILE = 'prediction_log.csv'

LOG_COLUMNS = [
    'timestamp', 'crop', 'condition', 'confidence',
    'latitude', 'longitude', 'image_name'
]


def load_log() -> pd.DataFrame:
    """Loads the existing prediction log or returns an empty frame."""
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_log(crop, condition, confidence, lat, lon, image_name):
    """Appends one prediction row to the log CSV."""
    df = load_log()
    new_row = pd.DataFrame([{
        'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'crop':        crop,
        'condition':   condition,
        'confidence':  round(confidence * 100, 2),
        'latitude':    lat if lat is not None else '',
        'longitude':   lon if lon is not None else '',
        'image_name':  image_name,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


# ================================================================
# GPS HELPER -- JavaScript geolocation via Streamlit component
# ================================================================

def get_gps_coordinates():
    """
    Attempts to get browser GPS coordinates using a hidden HTML/JS
    component. Returns (lat, lon) if available, or (None, None) if
    the user denies permission or the browser does not support it.

    The result is stored in st.session_state so it persists.
    """
    if 'gps_lat' not in st.session_state:
        st.session_state['gps_lat'] = None
        st.session_state['gps_lon'] = None

    try:
        from streamlit_js_eval import get_geolocation
        location = get_geolocation()
        if location and 'coords' in location:
            st.session_state['gps_lat'] = location['coords']['latitude']
            st.session_state['gps_lon'] = location['coords']['longitude']
    except Exception:
        pass   # streamlit-js-eval not available or permission denied

    return st.session_state['gps_lat'], st.session_state['gps_lon']


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.image(
        'https://em-content.zobj.net/source/apple/391/seedling_1f331.png',
        width=60
    )
    st.title('CropGuard AI')
    st.caption('Offline-first Crop Disease Detection')

    st.markdown('---')
    st.markdown('### Supported Crops')
    for crop_info in [
        ('🍅', 'Tomato',       '3 diseases'),
        ('🫑', 'Bell Pepper',  '1 disease'),
        ('🌽', 'Maize',        '2 diseases'),
        ('🥔', 'Potato',       '2 diseases'),
        ('🍇', 'Vine Crops',   '2 diseases'),
    ]:
        st.markdown(
            f"{crop_info[0]} **{crop_info[1]}** — {crop_info[2]}",
            unsafe_allow_html=False
        )

    st.markdown('---')
    st.markdown('### How to Use')
    st.markdown(
        '1. Upload a clear leaf photo\n'
        '2. Click **Analyse**\n'
        '3. Read the diagnosis\n'
        '4. Follow the treatment steps\n'
        '5. Log the case (optional)'
    )

    st.markdown('---')
    st.info(
        '**Offline-ready**: After the model loads, '
        'no internet connection is needed for predictions.',
        icon='📡'
    )

    st.markdown('---')
    st.caption('NextGen Innovation Challenge 2026 | Nigeria')

# ================================================================
# MAIN INTERFACE
# ================================================================

st.title('🌿 CropGuard AI')
st.subheader('Real-time Crop Disease Detection & Treatment Advisor')
st.markdown(
    'Take a photo of a crop leaf and receive an instant AI-powered '
    'diagnosis with actionable treatment advice — works offline.'
)

st.markdown('---')

# Load model
model = load_model()

if model is None:
    st.error(
        '**Model file not found.**\n\n'
        'Please ensure `crop_disease_model.h5` is in the same folder as '
        'this app. Run the training script (`train.py`) in Google Colab '
        'to generate it, then upload it to this Space.'
    )
    st.stop()

st.success('✅ AI model loaded and ready.')

# ----------------------------------------------------------------
# Image input section
# ----------------------------------------------------------------
st.markdown('### Step 1 — Provide a Leaf Image')

input_tab, camera_tab = st.tabs(['📁 Upload Image', '📷 Use Camera'])

uploaded_file = None
captured_file = None

with input_tab:
    uploaded_file = st.file_uploader(
        'Upload a JPG or PNG photo of a crop leaf',
        type=['jpg', 'jpeg', 'png'],
        help='For best results: clear photo, single leaf, natural lighting'
    )

with camera_tab:
    captured_file = st.camera_input(
        'Take a photo of the leaf',
        help='Position the leaf clearly in frame before capturing'
    )

# Determine the active image source
active_image_source = uploaded_file or captured_file
image_name = (
    uploaded_file.name if uploaded_file
    else 'camera_capture.jpg' if captured_file
    else None
)

# ----------------------------------------------------------------
# Prediction section
# ----------------------------------------------------------------
if active_image_source is not None:

    st.markdown('---')
    st.markdown('### Step 2 — Analysis')

    col_img, col_result = st.columns([1, 1], gap='large')

    with col_img:
        try:
            pil_image = Image.open(active_image_source)
            st.image(
                pil_image,
                caption=f'Input image: {image_name}',
                use_column_width=True
            )
        except Exception as exc:
            st.error(f'Could not open image: {exc}')
            st.stop()

    with col_result:
        with st.spinner('Analysing leaf...'):
            predicted_class, confidence, all_probs = run_prediction(
                model, pil_image
            )

        advice = TREATMENT_ADVICE.get(predicted_class, {})
        is_healthy = 'Healthy' in predicted_class
        urgency    = advice.get('urgency', '')

        # -- Diagnosis header --
        if is_healthy:
            st.success(f"**Diagnosis: {predicted_class}**")
        elif '🔴' in urgency:
            st.error(f"**Diagnosis: {predicted_class}**")
        elif '🟠' in urgency:
            st.warning(f"**Diagnosis: {predicted_class}**")
        else:
            st.warning(f"**Diagnosis: {predicted_class}**")

        # -- Confidence gauge --
        st.metric('Confidence', f'{confidence * 100:.1f}%')
        st.progress(float(confidence))

        if confidence < 0.60:
            st.caption(
                '⚠️ Low confidence. Please re-take the photo with '
                'better lighting or a closer view of the leaf.'
            )

        # -- Urgency --
        if urgency:
            st.markdown(f'**Urgency:** {urgency}')

        # -- Quick summary --
        if advice:
            st.markdown(f"**Cause:** {advice.get('cause', 'N/A')}")
            st.markdown(f"*{advice.get('symptoms', '')}*")

    # ----------------------------------------------------------------
    # Treatment advice section (full width)
    # ----------------------------------------------------------------
    if advice and not is_healthy:
        st.markdown('---')
        st.markdown('### Step 3 — Treatment Plan')

        with st.expander('📋 View Full Treatment Steps', expanded=True):
            steps = advice.get('treatment', [])
            for i, step in enumerate(steps, 1):
                st.markdown(f'**{i}.** {step}')

        col_prev, col_cost = st.columns(2)
        with col_prev:
            with st.expander('🛡️ Prevention Tips'):
                st.markdown(advice.get('prevention', 'N/A'))
        with col_cost:
            with st.expander('💰 Estimated Cost'):
                st.markdown(advice.get('cost_estimate', 'N/A'))

    elif is_healthy and advice:
        st.markdown('---')
        st.markdown('### Recommendations')
        with st.expander('📋 Maintenance Tips', expanded=True):
            for i, step in enumerate(advice.get('treatment', []), 1):
                st.markdown(f'**{i}.** {step}')

    # ----------------------------------------------------------------
    # Top-5 probability breakdown
    # ----------------------------------------------------------------
    st.markdown('---')
    st.markdown('### Probability Breakdown (Top 5)')

    top5_idx = np.argsort(all_probs)[::-1][:5]
    prob_data = {
        'Class':       [DISPLAY_NAMES[i] for i in top5_idx],
        'Probability': [f"{all_probs[i]*100:.1f}%" for i in top5_idx],
    }
    st.dataframe(
        pd.DataFrame(prob_data),
        hide_index=True,
        use_container_width=True
    )

    # ----------------------------------------------------------------
    # GPS logging section
    # ----------------------------------------------------------------
    st.markdown('---')
    st.markdown('### Step 4 — Log This Case (Optional)')

    log_col1, log_col2 = st.columns([2, 1])

    with log_col1:
        st.markdown(
            'Log this detection with GPS coordinates to track disease '
            'spread across your farm or region.'
        )
        enable_gps = st.checkbox('Enable GPS location', value=True)

        lat, lon = None, None

        if enable_gps:
            lat, lon = get_gps_coordinates()
            if lat is not None:
                st.success(f'📍 Location: {lat:.4f}°, {lon:.4f}°')
            else:
                st.info(
                    'GPS not available. You can enter coordinates manually below, '
                    'or log without location.'
                )
                man_col1, man_col2 = st.columns(2)
                with man_col1:
                    lat_input = st.text_input('Latitude', placeholder='e.g. 9.0765')
                with man_col2:
                    lon_input = st.text_input('Longitude', placeholder='e.g. 7.3986')
                if lat_input and lon_input:
                    try:
                        lat = float(lat_input)
                        lon = float(lon_input)
                    except ValueError:
                        st.error('Please enter valid decimal coordinates.')

    with log_col2:
        st.markdown(' ')
        st.markdown(' ')
        if st.button('💾 Save Detection to Log', use_container_width=True):
            crop_name = advice.get('crop', predicted_class.split(' - ')[0])
            condition = advice.get('condition', predicted_class.split(' - ')[-1])
            append_log(crop_name, condition, confidence, lat, lon, image_name)
            st.success('Detection logged successfully.')
            st.balloons()

# ================================================================
# DISEASE MAP SECTION
# ================================================================
st.markdown('---')
st.markdown('### 🗺️ Disease Case Map')

df_log = load_log()

if df_log.empty:
    st.info(
        'No cases logged yet. Analyse a leaf image and save it '
        'to see detections plotted on the map.'
    )
else:
    # Filter to rows that have valid coordinates
    df_map = df_log.dropna(subset=['latitude', 'longitude']).copy()
    df_map = df_map[
        (df_map['latitude'] != '') & (df_map['longitude'] != '')
    ]

    if df_map.empty:
        st.info('No cases with GPS coordinates found in the log yet.')
    else:
        df_map['latitude']  = pd.to_numeric(df_map['latitude'],  errors='coerce')
        df_map['longitude'] = pd.to_numeric(df_map['longitude'], errors='coerce')
        df_map = df_map.dropna(subset=['latitude', 'longitude'])

        if not df_map.empty:
            st.markdown(f'Showing **{len(df_map)}** logged detections with GPS coordinates.')
            st.map(
                df_map.rename(columns={'latitude': 'lat', 'longitude': 'lon'})[['lat', 'lon']],
                zoom=10
            )

    # Full log table
    with st.expander(f'📊 View All Logged Cases ({len(df_log)} total)'):
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        csv_data = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            '⬇️ Download Log as CSV',
            data=csv_data,
            file_name='cropguard_detections.csv',
            mime='text/csv',
            use_container_width=True
        )

# ================================================================
# ABOUT SECTION
# ================================================================
st.markdown('---')
with st.expander('ℹ️ About CropGuard AI'):
    st.markdown("""
**CropGuard AI** is an offline-first crop disease identification system
built for Nigerian farmers and agricultural extension workers.

**Model**: MobileNetV2 pretrained on ImageNet, fine-tuned on PlantVillage dataset  
**Classes**: 15 disease conditions across 5 crop types  
**Inference**: Runs entirely locally — no internet required after initial load  
**Edge deployment**: TFLite model available for Android/iOS integration  

**Crops covered:**
- 🍅 Tomato — Early Blight, Late Blight, Leaf Mold
- 🫑 Bell Pepper — Bacterial Spot
- 🌽 Maize — Common Rust, Northern Leaf Blight
- 🥔 Potato — Early Blight, Late Blight
- 🍇 Vine Crops — Black Rot, Black Measles

**Contact**: Submit queries via the NextGen Innovation Challenge 2026 portal.

*This tool supports farmers in making faster, better-informed decisions.  
Always consult a qualified agricultural extension officer for critical crop decisions.*
""")

st.markdown('---')
st.caption(
    'CropGuard AI — NextGen Innovation Challenge 2026 | '
    'Federal University Dutsin-Ma | Nigeria'
)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Moodify 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #141420;
    --accent: #c084fc;
    --accent2: #f472b6;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e1e30;
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text);
}

h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--text) !important; }

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Hero Section */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero h1 {
    font-size: 4rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 0;
}

/* Tab Style */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
}

/* Input */
.stTextInput > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(192, 132, 252, 0.2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Song Card */
.song-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s ease;
}
.song-card:hover { border-color: var(--accent); }
.song-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    min-width: 2rem;
    font-family: 'Playfair Display', serif;
}
.song-info h4 { margin: 0; font-size: 1rem; color: var(--text); }
.song-info p { margin: 2px 0 0; font-size: 0.85rem; color: var(--muted); }
.genre-badge {
    margin-left: auto;
    background: rgba(192, 132, 252, 0.15);
    color: var(--accent);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    white-space: nowrap;
}

/* Mood Chips */
.mood-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 1rem 0;
}
.mood-chip {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    cursor: pointer;
    display: inline-block;
}

/* Stats bar */
.stat-bar {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.stat-bar h3 { margin: 0; font-size: 1.8rem; color: var(--accent); }
.stat-bar p { margin: 0; font-size: 0.85rem; color: var(--muted); }

/* Selectbox */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* Slider */
.stSlider > div { color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ─── Load & Prepare Data ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    df = df.dropna(subset=features)
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

@st.cache_data
def build_similarity_matrix(df):
    features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    matrix = cosine_similarity(df[features])
    return matrix


# ─── Mood Engine ────────────────────────────────────────────────
MOOD_MAP = {
    # Happy / Energetic
    "happy":       {"valence": 0.85, "energy": 0.80, "danceability": 0.80, "acousticness": 0.10, "tempo": 0.75},
    "excited":     {"valence": 0.90, "energy": 0.95, "danceability": 0.90, "acousticness": 0.05, "tempo": 0.90},
    "joyful":      {"valence": 0.88, "energy": 0.78, "danceability": 0.82, "acousticness": 0.12, "tempo": 0.72},
    "celebrat":    {"valence": 0.92, "energy": 0.92, "danceability": 0.92, "acousticness": 0.05, "tempo": 0.88},
    "party":       {"valence": 0.90, "energy": 0.95, "danceability": 0.95, "acousticness": 0.03, "tempo": 0.92},
    "dance":       {"valence": 0.85, "energy": 0.90, "danceability": 0.95, "acousticness": 0.05, "tempo": 0.88},
    "fun":         {"valence": 0.88, "energy": 0.85, "danceability": 0.88, "acousticness": 0.08, "tempo": 0.80},

    # Sad / Lonely
    "sad":         {"valence": 0.15, "energy": 0.25, "danceability": 0.30, "acousticness": 0.80, "tempo": 0.30},
    "lonely":      {"valence": 0.12, "energy": 0.20, "danceability": 0.25, "acousticness": 0.85, "tempo": 0.25},
    "heartbreak":  {"valence": 0.10, "energy": 0.22, "danceability": 0.28, "acousticness": 0.82, "tempo": 0.28},
    "miss":        {"valence": 0.18, "energy": 0.28, "danceability": 0.32, "acousticness": 0.78, "tempo": 0.32},
    "depress":     {"valence": 0.08, "energy": 0.18, "danceability": 0.22, "acousticness": 0.88, "tempo": 0.22},
    "cry":         {"valence": 0.10, "energy": 0.20, "danceability": 0.25, "acousticness": 0.86, "tempo": 0.25},
    "broken":      {"valence": 0.12, "energy": 0.22, "danceability": 0.28, "acousticness": 0.84, "tempo": 0.28},

    # Romantic / Love
    "love":        {"valence": 0.75, "energy": 0.45, "danceability": 0.55, "acousticness": 0.65, "tempo": 0.50},
    "romantic":    {"valence": 0.78, "energy": 0.42, "danceability": 0.52, "acousticness": 0.68, "tempo": 0.48},
    "crush":       {"valence": 0.72, "energy": 0.48, "danceability": 0.58, "acousticness": 0.60, "tempo": 0.55},
    "date":        {"valence": 0.80, "energy": 0.50, "danceability": 0.60, "acousticness": 0.55, "tempo": 0.58},

    # Calm / Peaceful
    "calm":        {"valence": 0.60, "energy": 0.25, "danceability": 0.35, "acousticness": 0.85, "tempo": 0.30},
    "relax":       {"valence": 0.58, "energy": 0.22, "danceability": 0.32, "acousticness": 0.88, "tempo": 0.28},
    "peace":       {"valence": 0.62, "energy": 0.20, "danceability": 0.30, "acousticness": 0.90, "tempo": 0.25},
    "chill":       {"valence": 0.60, "energy": 0.28, "danceability": 0.40, "acousticness": 0.80, "tempo": 0.35},
    "sleep":       {"valence": 0.50, "energy": 0.10, "danceability": 0.20, "acousticness": 0.95, "tempo": 0.15},

    # Angry / Frustrated
    "angry":       {"valence": 0.20, "energy": 0.90, "danceability": 0.60, "acousticness": 0.05, "tempo": 0.90},
    "frustrat":    {"valence": 0.22, "energy": 0.85, "danceability": 0.55, "acousticness": 0.08, "tempo": 0.85},
    "stress":      {"valence": 0.25, "energy": 0.80, "danceability": 0.50, "acousticness": 0.10, "tempo": 0.80},

    # Motivated / Workout
    "motivat":     {"valence": 0.80, "energy": 0.92, "danceability": 0.75, "acousticness": 0.05, "tempo": 0.92},
    "workout":     {"valence": 0.75, "energy": 0.95, "danceability": 0.80, "acousticness": 0.03, "tempo": 0.95},
    "gym":         {"valence": 0.72, "energy": 0.95, "danceability": 0.78, "acousticness": 0.03, "tempo": 0.95},
    "focus":       {"valence": 0.55, "energy": 0.55, "danceability": 0.45, "acousticness": 0.60, "tempo": 0.55},
    "study":       {"valence": 0.52, "energy": 0.35, "danceability": 0.35, "acousticness": 0.75, "tempo": 0.40},

    # Nostalgic
    "nostalgi":    {"valence": 0.55, "energy": 0.40, "danceability": 0.45, "acousticness": 0.70, "tempo": 0.45},
    "memory":      {"valence": 0.52, "energy": 0.38, "danceability": 0.42, "acousticness": 0.72, "tempo": 0.42},
    "old":         {"valence": 0.58, "energy": 0.45, "danceability": 0.50, "acousticness": 0.65, "tempo": 0.48},

    # Default fallback
    "default":     {"valence": 0.55, "energy": 0.55, "danceability": 0.55, "acousticness": 0.45, "tempo": 0.55},
}

def detect_mood(text):
    text = text.lower()
    for keyword, vector in MOOD_MAP.items():
        if keyword != "default" and keyword in text:
            return vector, keyword
    return MOOD_MAP["default"], "neutral"

def recommend_by_mood(df, mood_vector, top_n=10, genre_filter=None):
    features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    query = np.array([[mood_vector[f] for f in features]])
    
    temp_df = df.copy()
    if genre_filter and genre_filter != "All":
        temp_df = temp_df[temp_df['track_genre'].str.lower() == genre_filter.lower()]
    
    if len(temp_df) == 0:
        temp_df = df.copy()

    scores = cosine_similarity(query, temp_df[features])[0]
    temp_df = temp_df.copy()
    temp_df['score'] = scores
    return temp_df.nlargest(top_n, 'score')[['track_name', 'artists', 'track_genre', 'score']].reset_index(drop=True)

def recommend_similar(df, song_name, sim_matrix, top_n=10):
    matches = df[df['track_name'].str.lower().str.contains(song_name.lower())]
    if matches.empty:
        return None, None
    
    idx = matches.index[0]
    song_data = df.iloc[idx]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = [s for s in scores if s[0] != idx][:top_n]
    top_df = df.iloc[[s[0] for s in top]][['track_name', 'artists', 'track_genre']].reset_index(drop=True)
    top_df['similarity'] = [round(s[1] * 100, 1) for s in top]
    return top_df, song_data


# ─── Song Card HTML ─────────────────────────────────────────────
def song_card(i, name, artist, genre, extra=""):
    return f"""
    <div class="song-card">
        <span class="song-num">{i}</span>
        <div class="song-info">
            <h4>{name}</h4>
            <p>{artist}</p>
        </div>
        <span class="genre-badge">{genre} {extra}</span>
    </div>"""


# ─── App Layout ─────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🎵 Moodify</h1>
        <p>Tell us how you feel — we'll find the perfect songs for your mood</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
        sim_matrix = build_similarity_matrix(df)
    except FileNotFoundError:
        st.error("❌ dataset.csv not found! Make sure you ran merge_datasets.py first.")
        return

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-bar"><h3>{len(df):,}</h3><p>Total Songs</p></div>', unsafe_allow_html=True)
    with col2:
        indian = df[df['track_genre'].isin(['bollywood','punjabi','tamil','telugu','sufi','indie','retro'])]
        st.markdown(f'<div class="stat-bar"><h3>{len(indian):,}</h3><p>Indian Songs</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-bar"><h3>{df["track_genre"].nunique()}</h3><p>Genres</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-bar"><h3>ML</h3><p>Cosine Similarity</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🧠 Mood Recommender", "🔍 Song Finder", "📊 How It Works"])

    # ── TAB 1: Mood Recommender ──────────────────────────────────
    with tab1:
        st.markdown("### How are you feeling right now?")

        # Quick mood chips (just UI labels)
        st.markdown("""
        <div class="mood-grid">
            <span class="mood-chip">😊 Happy</span>
            <span class="mood-chip">😢 Sad</span>
            <span class="mood-chip">❤️ Romantic</span>
            <span class="mood-chip">😤 Angry</span>
            <span class="mood-chip">🧘 Relaxed</span>
            <span class="mood-chip">💪 Motivated</span>
            <span class="mood-chip">🎉 Party</span>
            <span class="mood-chip">😔 Lonely</span>
            <span class="mood-chip">🏋️ Workout</span>
            <span class="mood-chip">📚 Study</span>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            mood_input = st.text_input(
                "Mood Input",
                placeholder='e.g. "I feel lonely today" or "I am so happy!"',
                label_visibility="collapsed"
            )
        with col_b:
            genre_options = ["All", "bollywood", "punjabi", "tamil", "telugu", "sufi", "indie", "retro"] + \
                            [g for g in df['track_genre'].unique() if g not in ["bollywood","punjabi","tamil","telugu","sufi","indie","retro"]]
            genre_filter = st.selectbox("Genre Filter", genre_options, label_visibility="collapsed")

        num_songs = st.slider("Number of songs", 5, 20, 10)

        if st.button("🎵 Get My Songs"):
            if mood_input.strip():
                mood_vector, detected = detect_mood(mood_input)
                st.success(f"Detected mood: **{detected.capitalize()}**")

                results = recommend_by_mood(df, mood_vector, top_n=num_songs, genre_filter=genre_filter)

                st.markdown(f"### Top {num_songs} songs for your mood")
                cards_html = ""
                for i, row in results.iterrows():
                    cards_html += song_card(i+1, row['track_name'], row['artists'], row['track_genre'])
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.warning("Please type how you're feeling!")

    # ── TAB 2: Song Finder ───────────────────────────────────────
    with tab2:
        st.markdown("### Find songs similar to one you love")
        
        col_s1, col_s2 = st.columns([4, 1])
        with col_s1:
            song_query = st.text_input(
                "Song search",
                placeholder='e.g. "Tum Hi Ho" or "Blinding Lights"',
                label_visibility="collapsed"
            )
        with col_s2:
            n_similar = st.selectbox("Show", [5, 8, 10, 15], index=2, label_visibility="collapsed")

        if st.button("🔍 Find Similar Songs"):
            if song_query.strip():
                results, song_info = recommend_similar(df, song_query, sim_matrix, top_n=n_similar)
                if results is not None:
                    st.success(f"Found similar songs to **{song_info['track_name']}** by {song_info['artists']}")
                    cards_html = ""
                    for i, row in results.iterrows():
                        extra = f"· {row['similarity']}% match"
                        cards_html += song_card(i+1, row['track_name'], row['artists'], row['track_genre'], extra)
                    st.markdown(cards_html, unsafe_allow_html=True)
                else:
                    st.error(f'No song found matching "{song_query}". Try a different name!')
            else:
                st.warning("Please enter a song name!")

    # ── TAB 3: How It Works ──────────────────────────────────────
    with tab3:
        st.markdown("### The ML Behind Moodify")
        
        st.markdown("""
        #### 🧮 Algorithm: Cosine Similarity

        Cosine Similarity measures the **angle between two feature vectors**. 
        Songs with similar audio features will have vectors pointing in the same direction — a high cosine score means high similarity.

        **Formula:**
        ```
        similarity = (A · B) / (||A|| × ||B||)
        ```

        #### 🎚️ Audio Features Used
        
        | Feature | What it measures |
        |---|---|
        | **Valence** | Musical positiveness (0 = sad, 1 = happy) |
        | **Energy** | Intensity and activity level |
        | **Danceability** | How suitable for dancing |
        | **Acousticness** | Amount of acoustic instruments |
        | **Tempo** | Speed of the track in BPM |

        #### 🔄 How Mood Recommendation Works

        1. You type your mood → keyword matching detects mood type
        2. Mood maps to a target feature vector (e.g. "sad" = low valence, high acousticness)
        3. Cosine similarity finds songs closest to that vector in 5D feature space
        4. Top N songs returned — these are your matches!

        #### 📊 Dataset
        - **{:,} total songs** across {:} genres
        - Combined Spotify global + Indian (Bollywood, Punjabi, Tamil, Telugu, Sufi, Indie, Retro)
        - Features normalized using MinMaxScaler before similarity computation
        """.format(len(df), df['track_genre'].nunique()))

        # Show sample data
        st.markdown("#### 📋 Sample from Dataset")
        sample = df[['track_name', 'artists', 'track_genre', 'valence', 'energy', 'danceability']].head(8)
        st.dataframe(sample, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

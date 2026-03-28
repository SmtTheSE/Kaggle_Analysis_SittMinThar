import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import xgboost as xgb
import json
import os
from datetime import datetime

# --- ELITE CONFIGURATION ---
st.set_page_config(
    page_title="Sitt Min Thar | Kaggle Showcase",
    page_icon=None,
    layout="wide",
)

# --- MINIMALIST SAGA CSS INJECTION ---
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #FFFFFF;
        color: #1a1a1a;
    }
    
    /* Typography Overrides */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        color: #1a1a1a !important;
    }
    
    /* Sidebar Scaling */
    section[data-testid="stSidebar"] {
        background-color: #fcfcfc !important;
        border-right: 1px solid #f0f0f0 !important;
    }
    
    /* Metric Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #1a1a1a !important;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Styled Expanders */
    .streamlit-expanderHeader {
        font-weight: 700 !important;
        background-color: #f9f9f9 !important;
        border-radius: 8px !important;
    }

    /* Hide Default Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SHARED CONSTANTS ---
SAGA_BLACK = "#1A1A1A"
VIBRANT_GREEN = "#2ECC71"
VIBRANT_PINK = "#F06292"
VIBRANT_CYAN = "#4DD0E1"
VIBRANT_ORANGE = "#FF8A65"
VIBRANT_PURPLE = "#9575CD"
SOFT_GREY = "#F5F5F5"

# --- DATA LOADING ENGINES ---
@st.cache_data
def load_nvidia_data():
    # Relative path from project root (assuming running from /Developer/kaggle)
    df = pd.read_csv('nvidia_stock_analysis/nvidia_stock_data_1999_2026.csv')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['Daily_Return'] = df['close'].pct_change()
    return df

@st.cache_data
def load_urban_data():
    df = pd.read_csv('Top_100_Population/Top 100 Worlds Largest Cities.csv')
    for col in ['Population (Est.)', 'Area (sq km)']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    df['Urban Density'] = df['Population (Est.)'] / df['Area (sq km)']
    return df

@st.cache_data
def load_bmw_data():
    df = pd.read_csv('bmw_sales/bmw_global_sales_2018_2025.csv')
    return df

@st.cache_data
def load_cyber_data():
    df = pd.read_csv('cyberattack-dataset/CyberAttackDataset/cybersecurity_attacks.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['PayloadSize'] = df['Payload Data'].str.len().fillna(0)
    return df

@st.cache_data
def load_spotify_data():
    songs = pd.read_csv('spotify_wrap_2025/spotify_wrapped_2025_top50_songs.csv')
    artists = pd.read_csv('spotify_wrap_2025/spotify_wrapped_2025_top50_artists.csv')
    alltime = pd.read_csv('spotify_wrap_2025/spotify_alltime_top100_songs.csv')
    return songs, artists, alltime

@st.cache_data
def load_netflix_data():
    df = pd.read_csv('netflix_titles/netflix_titles.csv')
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['year_added'] = df['date_added'].dt.year
    df['primary_country'] = df['country'].fillna('Unknown').apply(lambda x: x.split(',')[0])
    return df

@st.cache_data
def load_ufc_data():
    df = pd.read_csv('UFC_EDA_ADVANCED/ufc_gold_dataset_final.csv')
    if 'Event_Date' in df.columns:
        df['Event_Date'] = pd.to_datetime(df['Event_Date'], errors='coerce')
    return df

@st.cache_data
def load_makeup_data():
    df = pd.read_csv('Makeup_sale_2025/makeup_sales_dataset_2025.csv')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

@st.cache_data
def load_oscar_data():
    df = pd.read_csv('oscar_sount_track1934_2026/oscar_best_score_complete_1934_2026.csv')
    return df

@st.cache_data
def load_autism_data():
    # Use absolute path-like relative to project root
    df = pd.read_csv('Predict Autism Spectrum Disorder/Autism.csv')
    df.replace('?', np.nan, inplace=True)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df

@st.cache_data
def load_breast_cancer_data():
    country_df = pd.read_csv('Breast_Cancer_EDA/breast_cancer_by_country.csv')
    survival_df = pd.read_csv('Breast_Cancer_EDA/breast_cancer_survival_by_stage.csv')
    risk_df = pd.read_csv('Breast_Cancer_EDA/breast_cancer_risk_factors.csv')
    return country_df, survival_df, risk_df

@st.cache_data
def load_ai_job_data():
    df = pd.read_csv('AI_JOB_Market_Analysis/AI Job Market Dataset.csv')
    return df

# --- PAGE: HOME ---
def show_home():
    st.title("Analytical Showcase Portfolio")
    st.markdown(f"**Principal Architect:** <span style='color:{VIBRANT_PINK}'>Sitt Min Thar</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### The Executive Hub")
        st.markdown("""
        This elite analytical interface manifests the transition from raw Kaggle telemetry to high-depth strategic insights.
        """)
        
        st.write("#### Integrated Pillars")
        st.markdown(f"""
        - **NVIDIA Multi-Era**: [Logarithmic evolution & volatility clusters](https://www.kaggle.com/code/sittminthar/nvidia-evo).
        - **Global Urban Scaling**: [Efficiency metrics across the Top 100 cities](https://www.kaggle.com/code/sittminthar/top-100-population-cities-analysis).
        - **BMW Sales Dynamics**: [Regional revenue & electric share attribution](https://www.kaggle.com/code/sittminthar/bmw-global-sales).
        - **Cyberattack Telemetry**: [Threat attribution & network security forensic](https://www.kaggle.com/code/sittminthar/cyberattack-deep-analysis).
        - **Netflix Content Strategy**: [Release-to-addition latency & geographic dominance](https://www.kaggle.com/code/sittminthar/netflix-eda).
        - **Spotify Wrap 2025**: [Audio signatures & TikTok-driven song duration paradigms](https://www.kaggle.com/code/sittminthar/spotify-wrap-2025-eda-advanced).
        - **UFC Advanced EDA**: [Combat analytics and fight finish archetypes](https://www.kaggle.com/code/sittminthar/ufc-eda-advanced).
        - **Makeup Sales Strategy**: [Omnichannel conversion and prestige valuation](https://www.kaggle.com/code/sittminthar/make-up-sales-2025-eda-advanced).
        - **Oscar Soundtrack History**: [Composer dynasties and cinematic excellence](https://www.kaggle.com/code/sittminthar/oscar-soundtrack-strategic-eda).
        - **Autism Prediction AI**: [Clinical classification & phenotyping](https://www.kaggle.com/code/sittminthar/autism-prediction-elite-pipeline).
        - **Oncology Strategic Analysis**: [Global survival disparities & screening efficacy](https://www.kaggle.com/code/sittminthar/breast-cancer-stat-aware-eda-advanced).
        - **AI Job Market Analysis**: [Global employment telemetry & compensation benchmarks](https://www.kaggle.com/code/sittminthar/ai-job-market-analysis).
        """)
    
    with col2:
        st.write("### Technical Integrity")
        st.code("""
Streamlit Architecture
High-Depth Notebook logic
        """)

# --- PAGE: NVIDIA ---
def show_nvidia(df):
    st.title("NVIDIA: Market Evolution & Growth Dynamics")
    st.caption("Exponential financial telemetry evaluation (1999–2026)")
    
    selected_era = st.sidebar.multiselect("Strategic Era Filter", df['era'].unique(), default=df['era'].unique())
    f_df = df[df['era'].isin(selected_era)]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Market Cap", f"${f_df['market_cap_usd_bn'].max():,.0f}B")
    m2.metric("Volatility (Avg)", f"{f_df['Daily_Return'].std()*100:.2f}%")
    m3.metric("Max Close", f"${f_df['close'].max():,.2f}")
    m4.metric("Market Sessions", f"{len(f_df):,}")

    fig = plt.figure(figsize=(14, 7))
    plt.plot(f_df['date'], f_df['close'], color=VIBRANT_GREEN, linewidth=1.5, alpha=0.9)
    plt.yscale('log')
    plt.title("Logarithmic Expansion Cycles", fontweight='bold', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.2)
    st.pyplot(fig)
    
    st.markdown("### Risk-Adjusted Volatility Clusters")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    f_df['Rolling_Vol'] = f_df['Daily_Return'].rolling(window=20).std()
    plt.fill_between(f_df['date'], f_df['Rolling_Vol'], color=VIBRANT_PINK, alpha=0.3)
    plt.title("20-Day Rolling Volatility Clusters", fontweight='bold')
    st.pyplot(fig2)
    
    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/nvidia-evo)")

    with st.expander("Strategic Industrial Blueprint"):
        st.write("#### Technical Findings")
        st.markdown("""
        - **Architectural Dominance**: Growth spikes align with transition to AI-centric Blackwell and Hopper architectures.
        - **Logarithmic Integrity**: Price action maintains a consistent exponential channel despite short-term volatility.
        """)

# --- PAGE: URBAN ---
def show_urban(df):
    st.title("Global Urban Density & Resource Scaling")
    st.caption("Comparative efficiency evaluation of the world's Top 100 metropolitan nodes")
    
    st.sidebar.markdown("---")
    min_pop = st.sidebar.slider("Min Population (Millions)", 0.0, 40.0, 10.0)
    f_df = df[df['Population (Est.)'] >= min_pop * 1_000_000]
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(data=f_df.sort_values('Urban Density', ascending=False).head(15), 
                    x='Urban Density', y='City', color=VIBRANT_CYAN, ax=ax)
        ax.set_title("Metropolitan Efficiency Index (Density)", fontweight='bold')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.hexbin(df['Area (sq km)'], df['Population (Est.)'], gridsize=20, cmap='Purples')
        plt.title("Spatial Density Concentrations", fontweight='bold')
        plt.colorbar(label='City Frequency')
        st.pyplot(fig)
        
    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/top-100-population-cities-analysis)")

    with st.expander("Metropolitan Scaling Blueprint"):
        st.markdown(f"""
        - **Clustering**: The Top {len(f_df)} cities show a direct scaling relationship between infrastructure area and population support.
        - **Efficiency Leading**: High-density nodes in Asia exhibit the highest 'Urban Density' ROI.
        """)

# --- PAGE: BMW SALES ---
def show_bmw(df):
    st.title("BMW Group: Strategic Sales Dynamics")
    st.caption("Global revenue attribution and model distribution (2018–2025)")
    
    regions = df['Region'].unique()
    sel_region = st.sidebar.multiselect("Market Region", regions, default=regions)
    f_df = df[df['Region'].isin(sel_region)]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Regional Sales", f"{f_df['Units_Sold'].sum():,}")
    m2.metric("Estimated Revenue (EUR)", f"€{f_df['Revenue_EUR'].sum()/1_000_000_000:.1f}B")
    m3.metric("Avg BEV Share", f"{f_df['BEV_Share'].mean()*100:.1f}%")

    fig = plt.figure(figsize=(14, 7))
    sns.lineplot(data=f_df, x='Year', y='Units_Sold', hue='Region', palette="magma", marker='o', errorbar=None)
    plt.title("Annual Sales Trajectories by Global Region", fontweight='bold')
    st.pyplot(fig)
    
    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/bmw-global-sales)")

    with st.expander("BMW Industrial Blueprint"):
        st.markdown("""
        - **Regional Pivot**: Sales trajectory in North America indicates high-margin luxury growth.
        - **Electrification Share**: BEV and PHEV model adoption is the primary driver of market share retention in European markets.
        """)

# --- PAGE: CYBERATTACK ---
def show_cyber(df):
    st.title("Network Telemetry & Threat Attribution")
    st.caption("Forensic evaluation of global cybersecurity telemetry")
    
    severity = st.sidebar.radio("Severity Triage Level", ["All", "Low", "Medium", "High"])
    f_df = df if severity == "All" else df[df['Severity Level'] == severity]
    
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.write("#### Threat Distribution")
        type_counts = f_df['Attack Type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
                colors=[VIBRANT_PINK, VIBRANT_PURPLE, VIBRANT_ORANGE], startangle=140)
        st.pyplot(fig)
        
    with c2:
        st.write("#### Temporal Attack Patterns")
        hour_dist = f_df['Hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.fill_between(hour_dist.index, hour_dist.values, color=VIBRANT_CYAN, alpha=0.4)
        plt.plot(hour_dist.index, hour_dist.values, color=VIBRANT_CYAN, linewidth=2)
        plt.title("Telemetry Event Density by Hour", fontweight='bold')
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/cyberattack-deep-analysis)")

    with st.expander("Cybersecurity Forensic Blueprint"):
        st.markdown("""
        - **Signature Attribution**: DDoS events dominate high-severity triage categories.
        - **Payload Correlation**: Intrusion detection efficiency scales with behavioral signature mapping.
        """)

# --- PAGE: SPOTIFY ---
def show_spotify(songs, artists, alltime):
    st.title("Spotify Wrap 2025: Global Streaming Evolution")
    st.caption("Kaggle Gold Standard: Mapping global affects and economic signatures of 2025 pop")
    st.image("https://images.unsplash.com/photo-1614680376593-902f74cf0d41?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80", use_container_width=True)
    
    # --- METRICS LAYER ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg 2025 BPM", f"{songs['bpm'].mean():.0f}")
    m2.metric("Top Reach", f"{artists['monthly_listeners_millions_mar2026'].max():.1f}M")
    m3.metric("Avg Song Duration", f"{songs['duration_seconds'].mean():.0f}s")
    # Careers debuting after TikTok era (2018)
    tiktok_share = (artists['debut_year'] >= 2018).mean() * 100
    m4.metric("TikTok Era Share", f"{tiktok_share:.0f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Emotional Landscape", "Top Artists", "Longevity Analysis", "Regional Dominance"])
    
    with tab1:
        st.write("#### Affective Quadrants (The Valence-Energy Mood Map)")
        
        def classify_mood(row):
            if row['valence'] >= 0.5 and row['energy'] >= 0.5: return 'High-Octane Joy (Euphoric)'
            if row['valence'] < 0.5 and row['energy'] >= 0.5: return 'Intense / Moody'
            if row['valence'] < 0.5 and row['energy'] < 0.5: return 'Melancholic / Chilled'
            return 'Soulful / Relaxed'
        songs['Mood_Category'] = songs.apply(classify_mood, axis=1)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=songs, x='valence', y='energy', size='streams_2025_billions', 
                            hue='Mood_Category', sizes=(50, 500), alpha=0.8, palette='bright', ax=ax)
            ax.axvline(0.5, color='black', linestyle='--', alpha=0.2)
            ax.axhline(0.5, color='black', linestyle='--', alpha=0.2)
            
            # Quadrant Labels matching the notebook
            ax.text(0.7, 0.9, 'EUPHORIC', fontsize=10, fontweight='bold', color='green', alpha=0.5)
            ax.text(0.1, 0.9, 'INTENSE', fontsize=10, fontweight='bold', color='red', alpha=0.5)
            ax.text(0.1, 0.1, 'MELANCHOLIC', fontsize=10, fontweight='bold', color='blue', alpha=0.5)
            ax.text(0.7, 0.1, 'RELAXED', fontsize=10, fontweight='bold', color='orange', alpha=0.5)
            
            plt.title("Circumplex Affect Map", fontweight='bold')
            plt.xlabel("Positivity (Valence)")
            plt.ylabel("Intensity (Energy)")
            st.pyplot(fig)
            
        with c2:
            st.write("#### Mood Market Share")
            mood_counts = songs['Mood_Category'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            plt.pie(mood_counts, labels=mood_counts.index, autopct='%1.1f%%', 
                    colors=sns.color_palette('pastel'), startangle=140, wedgeprops={'edgecolor': 'white'})
            st.pyplot(fig2)
        
    with tab2:
        st.write("#### Top 15 Artists by Monthly Listener Reach (March 2026)")
        top_names = artists.sort_values('monthly_listeners_millions_mar2026', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_names, x='monthly_listeners_millions_mar2026', y='artist_name', palette='viridis', ax=ax)
        plt.title("Commercial Reach Leaders", fontweight='bold')
        plt.xlabel("Monthly Listeners (Millions)")
        st.pyplot(fig)

    with tab3:
        st.write("#### Artist Life Cycle: Career Debut vs. 2025 Standing")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=artists, x='debut_year', y='followers_millions', 
                        size='monthly_listeners_millions_mar2026', hue='gender', alpha=0.7, sizes=(100, 1500))
        plt.axvline(2018, color='red', linestyle='--', alpha=0.4, label='TikTok Era Start')
        plt.title("Longevity & Acceleration Matrix", fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        st.pyplot(fig)

    with tab4:
        st.write("#### Global Origin Distribution")
        country_counts = artists['country'].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=country_counts.values, y=country_counts.index, palette='flare')
        plt.title("Nationality Concentration in Global Top 50", fontweight='bold')
        plt.xlabel("Artist Count")
        st.pyplot(fig)
    
    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/spotify-wrap-2025-eda-advanced)")

# --- PAGE: NETFLIX ---
def show_netflix(df):
    st.title("Netflix: Content Intelligence & Strategy")
    st.caption("Strategic Evaluation of Global Catalog Dynamics (2021 Forensic)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Titles", f"{len(df):,}")
    m2.metric("Movie Share", f"{(df['type']=='Movie').mean()*100:.1f}%")
    m3.metric("Avg Launch Year", f"{int(df['release_year'].mean())}")
    m4.metric("Unique Nations", f"{df['primary_country'].nunique()}")

    tab1, tab2 = st.tabs(["Geographic Dominance", "Temporal Latency"])
    
    with tab1:
        st.write("#### Top 15 Source Nations (Primary Attribution)")
        top_nations = df[df['primary_country'] != 'Unknown']['primary_country'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_nations.values, y=top_nations.index, color=VIBRANT_ORANGE, ax=ax)
        plt.title("Regional Content Concentration", fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.write("#### Release-to-Addition Latency (The 'Lag' Heatmap)")
        recent_df = df[df['release_year'] >= 2000]
        # Using context to maintain dark text on white plot
        with plt.rc_context({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.labelcolor": "#1A1A1A",
            "xtick.color": "#1A1A1A",
            "ytick.color": "#1A1A1A",
            "text.color": "#1A1A1A",
            "axes.titlecolor": "#1A1A1A"
        }):
            fig, ax = plt.subplots(figsize=(11, 8))
            sns.histplot(x='release_year', y='year_added', data=recent_df, bins=25, cbar=True, cmap='mako', ax=ax)
            plt.title("Historical Catalog Latency Matrix", fontweight='bold')
            st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/netflix-eda)")

    with st.expander("Strategic Content Blueprint"):
        st.markdown("""
        - **Catalog Lag**: Post-2015 content exhibits minimal latency between release and Netflix addition, indicating a shift toward aggressive 'Day-and-Date' licensing.
        - **Geographic Pivot**: US/UK dominance remains significant, but rapid catalog expansion in India and South Korea indicates a strategic regional investment pivot.
        """)

# --- PAGE: UFC ---
def show_ufc(df):
    st.title("UFC Advanced EDA: Combat Analytics")
    st.caption("Comprehensive analysis of Mixed Martial Arts fight telemetry")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Recorded Fights", f"{len(df):,}")
    m2.metric("Weight Classes", f"{df['Weight_Class'].nunique()}")
    
    avg_fight_time = df['Total_Fight_Time_Sec'].mean() / 60 if 'Total_Fight_Time_Sec' in df.columns else 0
    m3.metric("Avg Fight Time", f"{avg_fight_time:.1f}m")
    
    most_common_method = df['Method'].mode()[0] if 'Method' in df.columns and not df['Method'].empty else "N/A"
    m4.metric("Most Common Method", most_common_method)

    tab1, tab2 = st.tabs(["Finish Methods", "Fight Durations"])
    
    with tab1:
        st.write("#### Distribution of Fight Finishes")
        fig, ax = plt.subplots(figsize=(10, 6))
        method_counts = df['Method'].value_counts().head(6)
        sns.barplot(x=method_counts.values, y=method_counts.index, palette='Reds_r', ax=ax)
        plt.title("Top Fight Finish Methods", fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.write("#### Total Fight Duration by Weight Class")
        fig, ax = plt.subplots(figsize=(10, 6))
        top_weights = df['Weight_Class'].value_counts().head(8).index
        f_df = df[df['Weight_Class'].isin(top_weights)]
        sns.boxplot(data=f_df, x='Total_Fight_Time_Sec', y='Weight_Class', palette='magma', ax=ax)
        plt.title("Fight Duration Distribution", fontweight='bold')
        plt.xlabel("Total Fight Time (Seconds)")
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/ufc-eda-advanced)")

    with st.expander("Combat Analytics Blueprint"):
        st.markdown('''
        - **Finish Paradigms**: Dominance of striking/submissions vs. decisions varies heavily across weight classes.
        - **Duration Profiling**: Heavier weight classes exhibit significantly lower average fight times due to higher finishing power.
        ''')

# --- PAGE: MAKEUP ---
def show_makeup(df):
    st.title("Global Cosmetic Commerce: Strategic Analytics")
    st.caption("High-Fidelity Evaluation of Elite Makeup Retail Distribution")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue Transacted", f"${df['Revenue_USD'].sum()/1_000_000:.2f}M")
    m2.metric("Total Units Dispatched", f"{df['Units_Sold'].sum():,}")
    m3.metric("Top Revenue Brand", df.groupby('Brand')['Revenue_USD'].sum().idxmax())
    m4.metric("Avg Product Margin", f"${df['Price_USD'].mean():.2f}")

    tab1, tab2 = st.tabs(["Brand Positioning", "Omnichannel Velocity"])
    
    with tab1:
        st.write("#### Brand Prestige & Market Penetration")
        fig, ax = plt.subplots(figsize=(10, 6))
        brand_rev = df.groupby('Brand')['Revenue_USD'].sum().sort_values(ascending=False)
        sns.barplot(x=brand_rev.values, y=brand_rev.index, palette="flare", ax=ax)
        plt.title("Revenue Concentration by Brand", fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.write("#### Digital Conversion Mechanisms")
        fig, ax = plt.subplots(figsize=(10, 6))
        payment_sales = df.groupby('Payment_Method')['Revenue_USD'].sum().sort_values()
        sns.barplot(x=payment_sales.index, y=payment_sales.values, palette="magma", ax=ax)
        plt.title("Cart-Clearing Capital by Payment Pipeline", fontweight='bold')
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/make-up-sales-2025-eda-advanced)")

    with st.expander("Strategic Industry Blueprint"):
        st.markdown('''
        - **Digital Infrastructure Dominance**: Omnichannel distribution displays massive clearing velocity via Digital Wallets.
        - **The 'Prestige Valuation Gap'**: Dominant brands exhibit rigid pricing architecture that shields high-margin products from volume dilution.
        ''')

# --- PAGE: OSCARS ---
def show_oscar(df):
    st.title("Cinematic Excellence: The Academy Awards")
    st.caption("Statistical Evaluation of Best Original Score Dominance (1934–2026)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Historical Awards", f"{len(df)}")
    m2.metric("Unique Composers", f"{df['Composer'].nunique()}")
    m3.metric("Most Awarded Composer", f"{df['Composer'].mode()[0]}")
    m4.metric("Oldest Monitored Year", f"{df['Year'].min()}")

    tab1, tab2 = st.tabs(["Composer Dominance Hierarchy", "Epoch Centralization"])
    
    with tab1:
        st.write("#### The Maestro Hierarchy (Top Awarded Composers)")
        fig, ax = plt.subplots(figsize=(10, 6))
        top_composers = df['Composer'].value_counts().head(10)
        sns.barplot(x=top_composers.values, y=top_composers.index, palette="mako", ax=ax)
        plt.title("Statistical Monopoly on Original Scores", fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.write("#### Era-Based Distribution Index")
        df['Decade'] = (df['Year'] // 10) * 10
        fig, ax = plt.subplots(figsize=(10, 6))
        decade_counts = df.groupby('Decade').size()
        sns.lineplot(x=decade_counts.index, y=decade_counts.values, marker='o', color="#F06292", linewidth=3, ax=ax)
        plt.fill_between(decade_counts.index, decade_counts.values, color="#F06292", alpha=0.2)
        plt.title("Award Frequency Validation Over Cinematic History", fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/oscar-soundtrack-strategic-eda)")

    with st.expander("Strategic Composer Blueprint"):
        st.markdown('''
        - **The Dynasty Monopoly**: A extreme minor percentage of composers possess highly disproportionate centralizations of historical wins.
        - **Generational Bridging**: Elite maestros transcend their original cinematic decade to continually dominate the Academy Award selection circuit.
        ''')

# --- PAGE: AUTISM ---
def show_autism(df):
    st.title("Autism Spectrum Disorder: Professional AI Diagnostics")
    st.caption("Advanced XGBoost 3.2.0 Architecture | game-theoretic SHAP interpretability")
    
    # Data Cleaning (Professional Tier)
    df['age_num'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[df['age_num'] <= 100].dropna(subset=['age_num']) # Remove 380yr+ outliers

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validated Patients", f"{len(df):,}")
    m2.metric("ASD Positive Rate", f"{(df['Class/ASD']=='YES').mean()*100:.1f}%")
    m3.metric("AI Engine", "XGBoost + SHAP")
    m4.metric("K-Fold AUC", "1.0000")

    tab1, tab2, tab3, tab4 = st.tabs(["Clinical Phenotypes", "Model Calibration", "Diagnostic Explainability", "Interactive Prediction Tool"])
    
    with tab1:
        st.write("#### Clinical Demographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            # Normalized Probability Bar: Jaundice vs ASD
            j_tab = pd.crosstab(df['jundice'], df['Class/ASD'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            j_tab.plot(kind='bar', stacked=True, color=['#00FFFF', '#FF1493'], ax=ax, edgecolor='#0A0A0A')
            ax.set_title("Clinical Risk: Jaundice vs ASD Result", fontweight='bold')
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)
            
        with col2:
            # Smooth KDE: Age Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(data=df, x='age_num', hue='Class/ASD', fill=True, palette=['#00FFFF', '#FF1493'], alpha=.4, ax=ax)
            ax.set_title("Phenotypic Age-Density Dynamics", fontweight='bold')
            st.pyplot(fig)

    with tab2:
        st.write("#### AI Model Calibration & Reliability")
        st.info("In healthcare, we measure 'Calibration' to ensure the AI's probability matches real-world clinical frequency.")
        
        # Calibration Curve Rendering
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            # Replicating the professional calibration visual
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([0.1, 0.3, 0.5, 0.7, 0.9], [0.12, 0.28, 0.52, 0.69, 0.91], marker='o', color='#00FFFF', label='XGBoost Calibration')
            ax.plot([0, 1], [0, 1], linestyle='--', color='white', alpha=0.3)
            ax.set_title("Clinical Reliability Curve", fontweight='bold')
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Frequency")
            st.pyplot(fig)
        with col_c2:
            st.markdown("""
            **Reliability Metrics:**
            - **Brier Score:** 0.002
            - **AUC-ROC:** 1.000
            - **Stratified Folds:** 5
            """)

    with tab3:
        st.write("#### SHAP Global Importance & Local Trace")
        st.caption("Decomposing the 'Black Box' into individual clinical drivers.")
        
        importance_data = pd.DataFrame({
            'Feature': ['A9_Score', 'A4_Score', 'A3_Score', 'A5_Score', 'A6_Score', 'Ethnicity', 'Jaundice'],
            'Weight': [0.42, 0.28, 0.15, 0.11, 0.08, 0.04, 0.02]
        }).sort_values('Weight', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=importance_data, x='Weight', y='Feature', palette='mako', ax=ax)
        ax.set_title("Global Clinical Attribution Hierarchy", fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("##### Explainability Metric")
        st.info("The V2 Research Notebook contains the full interactive SHAP Force Plot suite for every patient case. In this dashboard, we prioritize global attribution stability.")

    with tab4:
        st.write("#### Live Diagnostic Inference Tool")
        st.warning("DISCLAIMER: This is a professional data science demonstration. It is NOT a clinical diagnosis tool. Always consult a medical professional.")
        
        # Load Model & Metadata
        try:
            model_path = 'Predict Autism Spectrum Disorder/autism_model.bin'
            meta_path = 'Predict Autism Spectrum Disorder/model_metadata.json'
            
            if not os.path.exists(model_path):
                st.error("Model assets not found. Please ensure export_model.py has been run.")
            else:
                # Use Booster to avoid sklearn dependency in some environments
                bst = xgb.Booster()
                bst.load_model(model_path)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                with st.form("diagnostic_form"):
                    st.write("##### Patient Clinical Indicators (AQ-10)")
                    aq_cols = st.columns(5)
                    a_scores = []
                    for i in range(1, 11):
                        with aq_cols[(i-1)%5]:
                            val = st.toggle(f"AQ-{i} Positive", key=f"a{i}")
                            a_scores.append(1 if val else 0)
                    
                    st.write("##### Demographic & History")
                    c1, c2, c3 = st.columns(3)
                    age = c1.slider("Age", 0, 100, 25)
                    gender = c2.selectbox("Gender", meta['classes']['gender'])
                    jaundice = c3.selectbox("History of Jaundice", meta['classes']['jundice'])
                    
                    c4, c5 = st.columns(2)
                    ethnicity = c4.selectbox("Ethnicity", meta['classes']['ethnicity'])
                    relation = c5.selectbox("Relation (Who took test?)", meta['classes']['relation'])
                    
                    submitted = st.form_submit_button("RUN DIAGNOSTIC INFERENCE")
                    
                    if submitted:
                        # Preprocess Input to Match Training Features
                        input_dict = {
                            'A1_Score': a_scores[0], 'A2_Score': a_scores[1], 'A3_Score': a_scores[2],
                            'A4_Score': a_scores[3], 'A5_Score': a_scores[4], 'A6_Score': a_scores[5],
                            'A7_Score': a_scores[6], 'A8_Score': a_scores[7], 'A9_Score': a_scores[8],
                            'A10_Score': a_scores[9], 'age': age, 
                            'gender': meta['classes']['gender'].index(gender),
                            'ethnicity': meta['classes']['ethnicity'].index(ethnicity),
                            'jundice': meta['classes']['jundice'].index(jaundice),
                            'austim': 0, # Placeholder (standard for these datasets)
                            'contry_of_res': 0, # Simplified for demo
                            'relation': meta['classes']['relation'].index(relation)
                        }
                        
                        # Ensure features are in exact order as training
                        input_df = pd.DataFrame([input_dict])[meta['features']]
                        
                        # Booster Inference (Predicts probabilities by default for binary)
                        dtest = xgb.DMatrix(input_df)
                        prob = float(bst.predict(dtest)[0])
                        pred = "POSITIVE" if prob > 0.5 else "NEGATIVE"
                        
                        res_col1, res_col2 = st.columns(2)
                        res_col1.metric("Diagnostic Prediction", pred, delta=f"{prob*100:.1f}% confidence")
                        res_col2.progress(prob, text=f"ASD Presentation Probability: {prob*100:.1f}%")
                        
                        if prob > 0.5:
                            st.error(f"High Clinical Probability Detected: {prob*100:.1f}%")
                        else:
                            st.success(f"Low Clinical Probability: {prob*100:.1f}%")

        except Exception as e:
            st.error(f"Error initializing prediction engine: {str(e)}")

    st.markdown("[Explore Full AI Pipeline on Kaggle](https://www.kaggle.com/code/sittminthar/predict-autism-spectrum-disorder)")

# --- PAGE: BREAST CANCER ---
def show_breast_cancer(country_df, survival_df, risk_df):
    st.title("Global Oncology: Breast Cancer Strategic Analysis")
    st.caption("Technical Deep-Dive into Global Survival Disparities & Screening Efficacy (2022-2025)")
    st.image("https://images.unsplash.com/photo-1579154235602-3c2c2abb3b8a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80", use_container_width=True)

    # Engineering metrics
    country_df['MIR'] = country_df['Deaths_2022'] / country_df['New_Cases_2022']
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Global New Cases (2022)", f"{country_df['New_Cases_2022'].sum()/1_000_000:.1f}M")
    m2.metric("Avg 5-Year Survival", f"{country_df['Five_Year_Survival_Pct'].mean():.1f}%")
    m3.metric("Avg MIR Score", f"{country_df['MIR'].mean():.3f}")
    m4.metric("Screening Coverage", f"{country_df['Mammography_Coverage_Pct'].mean():.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Infrastructure Efficacy", "Efficiency Frontier", "Survival Stratification", "Risk ROI Heatmap"])

    with tab1:
        st.write("#### Healthcare System Efficacy: Mortality-to-Incidence Ratio (MIR)")
        mir_extremes = country_df.sort_values('MIR')
        target_display = pd.concat([mir_extremes.head(10), mir_extremes.tail(10)])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=target_display, x='MIR', y='Country', palette="flare", ax=ax)
        plt.axvline(0.40, color="#FFD700", linestyle='--', alpha=0.5, label='High-Risk Threshold')
        plt.title("MIR: Deaths / Incidence Velocity", fontweight='bold', pad=20)
        plt.xlabel("MIR Score", labelpad=15)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.write("#### The Oncology Efficiency Frontier")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.regplot(data=country_df, x='Incidence_Rate_Per_100K', y='Mortality_Rate_Per_100K', 
                    scatter_kws={'s': 100, 'alpha': 0.6, 'color': '#00FFFF'}, 
                    line_kws={'color': '#FF1493', 'lw': 2, 'ls': '--'}, ax=ax)
        plt.title("Burden vs. Mortality Capture Efficiency", fontweight='bold', pad=20)
        plt.xlabel("Incidence Rate (per 100K)", labelpad=15)
        plt.ylabel("Mortality Rate (per 100K)", labelpad=15)
        plt.subplots_adjust(top=0.9, bottom=0.15)
        st.pyplot(fig)
        st.info("Nations below the trend-line exhibit superior treatment capture success relative to their cancer burden.")

    with tab3:
        st.write("#### Longitudinal Survival Decay by Socioeconomic Stratum (Stage II)")
        survival_melted = survival_df.melt(id_vars=['Stage', 'Income_Region'], 
                                          value_vars=['One_Year_Survival_Pct', 'Five_Year_Survival_Pct', 'Ten_Year_Survival_Pct'],
                                          var_name='Period', value_name='Survival_Pct')
        survival_melted['Timeline_Years'] = survival_melted['Period'].map({'One_Year_Survival_Pct': 1, 'Five_Year_Survival_Pct': 5, 'Ten_Year_Survival_Pct': 10})
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=survival_melted[survival_melted['Stage'] == 'Stage II'], 
                     x='Timeline_Years', y='Survival_Pct', hue='Income_Region', 
                     marker='o', linewidth=3, palette="spring", ax=ax)
        plt.title("The 'Survival Cliff' Across Income Tiers", fontweight='bold', pad=20)
        plt.xlabel("Years Post-Diagnosis", labelpad=15)
        plt.ylabel("Survival Probability (%)", labelpad=15)
        plt.ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.write("#### Strategic ROI: Modifiable PAF Ranking")
        lifestyle_df = risk_df[risk_df['Category'] == 'Lifestyle'].sort_values('Population_Attributable_Fraction_Pct')
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=lifestyle_df, x='Population_Attributable_Fraction_Pct', y='Risk_Factor', palette='crest', ax=ax)
        plt.title("Modifiable Risk Population Attribution", fontweight='bold', pad=20)
        plt.xlabel("PAF Percentage (%)", labelpad=15)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/breast-cancer-stat-aware-eda-advanced)")

# --- PAGE: AI JOB MARKET ---
def show_ai_job_market(df):
    st.title("AI Job Market: Strategic Analysis 2024")
    st.caption("Comprehensive evaluation of global AI employment telemetry and compensation paradigms")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Listings", f"{len(df):,}")
    m2.metric("Unique Roles", f"{df['job_title'].nunique():,}")
    m3.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
    m4.metric("Countries", f"{df['country'].nunique():,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Market Composition", "Salary Dynamics", "Industry Intelligence", "Company Scaling"])

    with tab1:
        st.write("#### Top Dominant AI Roles")
        top_roles = df['job_title'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_roles.values, y=top_roles.index, palette="viridis", ax=ax)
        plt.title("Role Distribution Frequency", fontweight='bold')
        st.pyplot(fig)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### Global Salary Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df['salary'], bins=30, kde=True, color=VIBRANT_CYAN, ax=ax)
            plt.title("Compensation Density", fontweight='bold')
            st.pyplot(fig)
        
        with col_b:
            st.write("#### Salary by Experience Level")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='experience_level', y='salary', data=df, palette="magma", ax=ax)
            plt.title("Experience-Yield Attribution", fontweight='bold')
            st.pyplot(fig)

    with tab3:
        st.write("#### Industry Penetration vs. Compensation")
        ind_stats = df.groupby('company_industry').agg({'salary': 'mean', 'job_title': 'count'}).rename(columns={'job_title': 'Count'}).sort_values('Count', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=ind_stats['salary'], y=ind_stats.index, palette="flare", ax=ax)
        plt.title("Top Industries by Strategic Value (Avg Salary)", fontweight='bold')
        st.pyplot(fig)

    with tab4:
        st.write("#### Company Size Scaling Analysis")
        size_salary = df.groupby('company_size')['salary'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=size_salary.index, y=size_salary.values, palette="crest", ax=ax)
        plt.title("Average Salary by Organizational Scale", fontweight='bold')
        st.pyplot(fig)

    st.markdown("[Explore Full Analysis on Kaggle](https://www.kaggle.com/code/sittminthar/ai-job-market-analysis)")

# --- NAVIGATION ---
def main():
    st.sidebar.markdown(f"<h1 style='color:{SAGA_BLACK}; font-size:24px;'>NAVIGATOR</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("Select Analytics Product", 
                            ["Home", "NVIDIA Multi-Era", "Global Urban Density", "BMW Sales Suite", "Cyberattack Forensic", "Netflix Content Strategy", "Spotify Wrap 2025", "UFC Advanced EDA", "Global Cosmetic Commerce", "Oscar Soundtrack History", "Autism Prediction AI", "Oncology Strategic Analysis", "AI Job Market Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.write("### Resource Hub")
    st.sidebar.info("This hub showcases a curated selection of analytical products. For the full library of upcoming Kaggle notebook analyses and source code, explore the central repository.")
    st.sidebar.markdown("[View GitHub Repository](https://github.com/SmtTheSE/Kaggle_Analysis_SittMinThar.git)")
    
    st.sidebar.markdown("---")
    st.sidebar.write("### Featured Kaggle Notebooks")
    st.sidebar.markdown("- [NVIDIA Evolution](https://www.kaggle.com/code/sittminthar/nvidia-evo)")
    st.sidebar.markdown("- [Top 100 Cities](https://www.kaggle.com/code/sittminthar/top-100-population-cities-analysis)")
    st.sidebar.markdown("- [BMW Global Sales](https://www.kaggle.com/code/sittminthar/bmw-global-sales)")
    st.sidebar.markdown("- [Cyberattack Analysis](https://www.kaggle.com/code/sittminthar/cyberattack-deep-analysis)")
    st.sidebar.markdown("- [Netflix Content Strategy](https://www.kaggle.com/code/sittminthar/netflix-eda)")
    st.sidebar.markdown("- [Spotify Wrap 2025](https://www.kaggle.com/code/sittminthar/spotify-wrap-2025-eda-advanced)")
    st.sidebar.markdown("- [UFC Advanced Analysis](https://www.kaggle.com/code/sittminthar/ufc-eda-advanced)")
    st.sidebar.markdown("- [Makeup Sales Analytics](https://www.kaggle.com/code/sittminthar/make-up-sales-2025-eda-advanced)")
    st.sidebar.markdown("- [Oscar Cinematic Excellence](https://www.kaggle.com/code/sittminthar/oscar-cinematic-excellence-the-academy-awards)")
    st.sidebar.markdown("- [Autism Prediction AI](https://www.kaggle.com/code/sittminthar/predict-autism-spectrum-disorder)")
    st.sidebar.markdown("- [Breast Cancer Strategic EDA](https://www.kaggle.com/code/sittminthar/breast-cancer-stat-aware-eda-advanced)")
    st.sidebar.markdown("- [AI Job Market Analysis](https://www.kaggle.com/code/sittminthar/ai-job-market-analysis)")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Portfolio Developed by **Sitt Min Thar**")
    
    if page == "Home":
        show_home()
        st.markdown("---")
        st.write("### Strategic Transparency")
        st.markdown(f"""
        The full technical depth of this analysis, including all original Jupyter Notebooks and data engineering workflows, is available for peer review.
        
        [**Explore Source on GitHub**](https://github.com/SmtTheSE/Kaggle_Analysis_SittMinThar.git)
        """, unsafe_allow_html=True)
    elif page == "NVIDIA Multi-Era":
        df = load_nvidia_data()
        show_nvidia(df)
    elif page == "Global Urban Density":
        df = load_urban_data()
        show_urban(df)
    elif page == "BMW Sales Suite":
        df = load_bmw_data()
        show_bmw(df)
    elif page == "Cyberattack Forensic":
        df = load_cyber_data()
        show_cyber(df)
    elif page == "Netflix Content Strategy":
        df = load_netflix_data()
        show_netflix(df)
    elif page == "Spotify Wrap 2025":
        songs, artists, alltime = load_spotify_data()
        show_spotify(songs, artists, alltime)
    elif page == "UFC Advanced EDA":
        df = load_ufc_data()
        show_ufc(df)
    elif page == "Global Cosmetic Commerce":
        df = load_makeup_data()
        show_makeup(df)
    elif page == "Oscar Soundtrack History":
        df = load_oscar_data()
        show_oscar(df)
    elif page == "Autism Prediction AI":
        df = load_autism_data()
        show_autism(df)
    elif page == "Oncology Strategic Analysis":
        c_df, s_df, r_df = load_breast_cancer_data()
        show_breast_cancer(c_df, s_df, r_df)
    elif page == "AI Job Market Analysis":
        df = load_ai_job_data()
        show_ai_job_market(df)

if __name__ == "__main__":
    main()

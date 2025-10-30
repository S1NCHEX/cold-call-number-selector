"""
Cold-call Number Selector with Streamlit Web UI

This Streamlit app replicates the behavior of your cold-calling assistant.
You can enter a target county/state (e.g., "Parker County, Texas"), and it will rank
and display the best numbers from your phone pool. It also supports uploading a
replacement CSV file of numbers directly in the web interface.

Deploy on Streamlit Cloud for a free shareable link.
"""

import streamlit as st
import pandas as pd
import re
import unicodedata
from typing import List, Dict, Tuple

# ----------------------------- Utilities -----------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9 ,.-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_area_code(phone: str) -> str:
    if not phone:
        return ""
    m = re.search(r"\+?1?\D*(\d{3})\D*\d{3}\D*\d{4}", phone)
    return m.group(1) if m else ""

CENSUS_REGIONS = {
    'northeast': {'maine','new hampshire','vermont','massachusetts','rhode island','connecticut','new york','new jersey','pennsylvania'},
    'midwest': {'ohio','michigan','indiana','illinois','wisconsin','minnesota','iowa','missouri','north dakota','south dakota','nebraska','kansas'},
    'south': {'delaware','maryland','district of columbia','virginia','west virginia','north carolina','south carolina','georgia','florida','kentucky','tennessee','mississippi','alabama','oklahoma','texas','arkansas','louisiana'},
    'west': {'montana','idaho','wyoming','nevada','utah','colorado','arizona','new mexico','alaska','washington','oregon','california','hawaii'},
}

def guess_region_from_state(state_name: str) -> str:
    s = normalize_text(state_name)
    for region, states in CENSUS_REGIONS.items():
        if s in states:
            return region
    return 'unknown'

def simple_label_similarity(a: str, b: str) -> float:
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    common = a_words & b_words
    return len(common) / max(len(a_words), len(b_words))

STATE_ABBREVS = {
    'al':'alabama','ak':'alaska','az':'arizona','ar':'arkansas','ca':'california','co':'colorado','ct':'connecticut','de':'delaware','fl':'florida','ga':'georgia','hi':'hawaii','id':'idaho','il':'illinois','in':'indiana','ia':'iowa','ks':'kansas','ky':'kentucky','la':'louisiana','me':'maine','md':'maryland','ma':'massachusetts','mi':'michigan','mn':'minnesota','ms':'mississippi','mo':'missouri','mt':'montana','ne':'nebraska','nv':'nevada','nh':'new hampshire','nj':'new jersey','nm':'new mexico','ny':'new york','nc':'north carolina','nd':'north dakota','oh':'ohio','ok':'oklahoma','or':'oregon','pa':'pennsylvania','ri':'rhode island','sc':'south carolina','sd':'south dakota','tn':'tennessee','tx':'texas','ut':'utah','vt':'vermont','va':'virginia','wa':'washington','wi':'wisconsin','wv':'west virginia','wy':'wyoming'
}

ALL_STATES = set(STATE_ABBREVS.values())

def guess_state_from_label(label: str) -> str:
    s = normalize_text(label)
    m = re.search(r"\b([A-Za-z]{2})\b", s)
    if m:
        ab = m.group(1).lower()
        if ab in STATE_ABBREVS:
            return STATE_ABBREVS[ab]
    for st in ALL_STATES:
        if st in s:
            return st
    CITY_HINTS = {
        'milwaukee':'wisconsin','madison':'wisconsin','appleton':'wisconsin','cleveland':'ohio','cincinnati':'ohio','toledo':'ohio','houston':'texas','dallas':'texas','amarillo':'texas','fort worth':'texas','austin':'texas','san francisco':'california','los angeles':'california','seattle':'washington','denver':'colorado','phoenix':'arizona','orlando':'florida','miami':'florida'
    }
    for city, st in CITY_HINTS.items():
        if city in s:
            return st
    return ''

def guess_county_from_label(label: str) -> str:
    s = normalize_text(label)
    m = re.search(r"([a-z '-]+) county", s)
    if m:
        return m.group(1).strip()
    tokens = s.split()
    if len(tokens) >= 2 and tokens[1] in STATE_ABBREVS.values():
        return tokens[0]
    return ''

def score_entry(target_label: str, target_state: str, target_county: str, target_area_code: str, entry: Dict) -> float:
    label = entry.get('label','')
    state = entry.get('state') or guess_state_from_label(label)
    county = entry.get('county') or guess_county_from_label(label)
    area = entry.get('area_code') or extract_area_code(entry.get('phone',''))

    score = 0.0
    if target_county and county and normalize_text(target_county) == normalize_text(county):
        score += 50
    if target_state and state and normalize_text(target_state) == normalize_text(state):
        score += 40
    if target_area_code and area and target_area_code == area:
        score += 35

    targ_region = guess_region_from_state(target_state or '')
    ent_region = guess_region_from_state(state or '')
    if targ_region != 'unknown' and targ_region == ent_region:
        score += 12

    score += 20 * simple_label_similarity(target_label, label)

    if entry.get('phone','').strip().startswith('+') and not entry.get('phone','').strip().startswith('+1'):
        score -= 10
    return score

def select_best_numbers(target: str, pool: List[Dict], n: int = 3) -> List[Tuple[str, float, Dict]]:
    targ_norm = normalize_text(target)
    targ_state = guess_state_from_label(targ_norm)
    targ_county = ''
    m = re.search(r"([a-z '-]+) county", targ_norm)
    if m:
        targ_county = m.group(1).strip()
    targ_area = ''
    m2 = re.search(r"\b(\d{3})\b", targ_norm)
    if m2:
        targ_area = m2.group(1)

    scored = []
    for entry in pool:
        sc = score_entry(targ_norm, targ_state, targ_county, targ_area, entry)
        scored.append((entry.get('phone',''), sc, entry))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]

# ----------------------------- Streamlit Web UI -----------------------------

def load_default_pool() -> List[Dict]:
    return [
        {'phone': '+16086846788', 'label': 'Dane County, WI'},
        {'phone': '+12624060854', 'label': 'Wisconsin'},
        {'phone': '+14143125720', 'label': 'Milwaukee County, WI'},
        {'phone': '+12109341449', 'label': 'Texas Mastersheet'},
        {'phone': '+12143909687', 'label': 'Collin County, TX'},
        {'phone': '+14234829877', 'label': 'Cocke County, TN'},
        {'phone': '+13156303846', 'label': 'Hamilton County, IN'},
    ]

st.set_page_config(page_title="Cold Call Number Selector", page_icon="📞", layout="centered")
st.title("📞 Cold Call Number Selector")
st.write("Upload your phone number pool or use the default. Then enter a county/state to find the top 3 most local numbers.")

uploaded_file = st.file_uploader("Upload your CSV (must have at least 'phone' and 'label' columns)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    pool = df.to_dict('records')
    st.success(f"Loaded {len(pool)} numbers from uploaded CSV.")
else:
    pool = load_default_pool()
    st.info(f"Using default pool of {len(pool)} numbers.")

target = st.text_input("Enter target area (e.g., 'Floyd County, Kentucky'):")

if st.button("Find Best Numbers"):
    if not target.strip():
        st.warning("Please enter a target area first.")
    else:
        results = select_best_numbers(target, pool, n=3)
        if not results:
            st.error("No matches found.")
        else:
            st.subheader(f"Top 3 Numbers for {target}:")
            for rank, (phone, score, entry) in enumerate(results, 1):
                st.markdown(f"### {rank}. {phone}")
                st.write(f"**Label:** {entry.get('label','')}  |  **Score:** {score:.1f}")
                state = guess_state_from_label(entry.get('label',''))
                county = guess_county_from_label(entry.get('label',''))
                if state or county:
                    st.caption(f"Guessed: {county.title() if county else ''} {('County' if county else '')} {state.title() if state else ''}")

st.divider()
st.caption("Developed for NSR Cold Calling Agent – Streamlit edition.")

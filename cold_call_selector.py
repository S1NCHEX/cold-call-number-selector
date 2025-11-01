"""
Cold-call Number Selector with Streamlit Web UI

- Uses YOUR CSV as the default phone pool (label, phone, notes).
- Lets you upload a replacement CSV (must include 'label' and 'phone'; 'notes' optional).
- Ranks the best numbers for a target county/state/city.
- Adds a toggleable timezone table at the bottom that shows local times
  for selected regions compared to EST (anchor).

Deploy on Streamlit Community Cloud for a free shareable link.
"""

import io
import re
import unicodedata
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

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
        'milwaukee':'wisconsin','madison':'wisconsin','appleton':'wisconsin',
        'cleveland':'ohio','cincinnati':'ohio','toledo':'ohio',
        'houston':'texas','dallas':'texas','amarillo':'texas','fort worth':'texas','austin':'texas',
        'san francisco':'california','los angeles':'california',
        'seattle':'washington','denver':'colorado','phoenix':'arizona',
        'orlando':'florida','miami':'florida'
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
        # Non-US numbers: slight penalty (still selectable)
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

import io
import pandas as pd

# ---- Your exact CSV, quoted ----
DEFAULT_CSV = """label,phone,notes
"Detroit","+13136313481","User not assigned; Local"
"Orlando, florida","+16893198987","User not assigned; Local"
"Nick vansons numbe...","+13156503380","Nick Vanson; Local"
"minneapolis","+16515151023","User not assigned; Local"
"nebraska (omaha)","+13082448081","User not assigned; Local"
"Washington DC","+17719998160","User not assigned; Local"
"San Francisco","+16282854335","User not assigned; Local"
"Toronto","+14378004745","User not assigned; Local"
"Ottawa","+16135050531","User not assigned; Local"
"Toledo","+14194064175","User not assigned; Local"
"Fort wayne","+12602352413","User not assigned; Local"
"Albaquerque","+15053913261","User not assigned; Local"
"Indianapolis","+14632639298","User not assigned; Local"
"Hampton roads numb...","+17574531289","User not assigned; Local"
"baltimore","+14437261916","User not assigned; Local"
"dallas TX","+14697784845","User not assigned; Local"
"philadelphia","+12674407571","User not assigned; Local"
"houston TX","+17138047181","User not assigned; Local"
"portland number","+15032075738","Nick Winson; Local"
"Vancouver","+17787290861","User not assigned; Local"
"las Vaages","+17026236093","User not assigned; Local"
"Indianapolis","+13174340379","User not assigned; Local"
"San Francisco","+14159121311","User not assigned; Local"
"Memphis","+19016271501","User not assigned; Local"
"Sacramento","+19162833614","User not assigned; Local"
"New York (amiro)","+16462120095","User not assigned; Local"
"Atlanta","+14049620885","User not assigned; Local"
"Stockton","+12093205848","User not assigned; Local"
"Chicago old","+17733789049","User not assigned; Local"
"Aferdita usa numbe...","+12137704748","User not assigned; Local"
"Afrodita Uk Number","+447700162154","User not assigned; Mobile"
"Phoenix amiro","+16233043702","User not assigned; Local"
"san diago old","+16193562991","User not assigned; Local"
"312 area code","+13126675871","User not assigned; Local"
"Seattle","+12062101847","User not assigned; Local"
"orlando","+13213207886","User not assigned; Local"
"Connecticut","+12039024478","User not assigned; Local"
"frenso 18/7","+15593963456","User not assigned; Local"
"CA - Joe - 7-18-25","+15304860375","User not assigned; Local"
"Bristol County Ami...","+15082995793","User not assigned; Local"
"California","+12133204957","User not assigned; Local"
"Windham county","+18603294737","User not assigned; Local"
"Joe's number 2","+16065521538","User not assigned; Local"
"ALPIN 7/22","+18585440943","User not assigned; Local"
"Joe's number 4 - C...","+18189228530","User not assigned; Local"
"San Bernardino Cou...","+19096464823","User not assigned; Local"
"San Mateo County, ...","+16505358166","User not assigned; Local"
"brooklyn","+19297327808","User not assigned; Local"
"Joe's 408 Santa Cl...","+14082908118","User not assigned; Local"
"Isabella Uk Number","+447723453492","User not assigned; Mobile"
"Isabella Usa Numbe...","+13108613523","Isabella matinez; Local"
"Amiro Erie County ...","+17163033523","User not assigned; Local"
"Sonoma county","+17072101857","User not assigned; Local"
"Ventura county","+18053073730","User not assigned; Local"
"san benito- 28july","+18312025936","User not assigned; Local"
"Joe's 7-28 Orange ...","+16575675630","User not assigned; Local"
"Middlesex county","+16176225452","User not assigned; Local"
"Spokane County Was...","+15094729349","Nick Amiro; Local"
"oklahoma city","+14055914806","User not assigned; Local"
"baldwin county, al","+12512572434","User not assigned; Local"
"Thurston County","+13603479540","User not assigned; Local"
"philadelphia","+12157701587","User not assigned; Local"
"houston county","+13344014356","User not assigned; Local"
"Cincinnati","+15135401641","User not assigned; Local"
"Monroe County, New...","+15855342759","User not assigned; Local"
"Jefferson County,A...","+12058786778","User not assigned; Local"
"Fiona's Number UK","+447361582425","Fiona Kallari; Mobile"
"Mariam-Craighead C...","+18705399733","User not assigned; Local"
"Ulster County, New...","+18456057742","User not assigned; Local"
"Washington County,...","+14793223858","User not assigned; Local"
"Cleveland Amiro","+12164005705","User not assigned; Local"
"Tioga County,New Y...","+16072282141","User not assigned; Local"
"saline county, ar","+15013009637","User not assigned; Local"
"Louisville Amiro","+15024026763","User not assigned; Local"
"Denver County, Col...","+17207069337","User not assigned; Local"
"Schoharie County –...","+15183274283","User not assigned; Local"
"Jefferson County, ...","+13035029795","User not assigned; Local"
"Adams county and o...","+17196243985","User not assigned; Local"
"Miami-Dade County,...","+17864811343","User not assigned; Local"
"weld and larimer c...","+19705146443","User not assigned; Local"
"Pima County, Arizo...","+15203359460","User not assigned; Local"
"kansas master shee...","+19133477112","User not assigned; Local"
"Maricopa County, A...","+16025841755","User not assigned; Local"
"Palm Beach County,...","+15615934995","User not assigned; Local"
"Arizona Master She...","+19283703498","User not assigned; Local"
"Gem County, Idaho","+19862164948","User not assigned; Local"
"Hawaii Master Shee...","+18084603743","User not assigned; Local"
"Broward County, Fl...","+19542874805","User not assigned; Local"
"Baltimore County, ...","+14107557904","User not assigned; Local"
"Hazem's number Flo...","+12393748726","User not assigned; Local"
"Duval County, Flor...","+19048306636","User not assigned; Local"
"alachua","+13863567791","User not assigned; Local"
"Hillsborough Count...","+18136925881","User not assigned; Local"
"Manatee County - F...","+19412026172","User not assigned; Local"
"Gooding County, Id...","+12085652789","User not assigned; Local"
"Escambia county, F...","+18506054421","User not assigned; Local"
"Jefferson Parish, ...","+15046051384","User not assigned; Local"
"St Lucie's County ...","+17723031168","User not assigned; Local"
"illinois master sh...","+12176346731","User not assigned; Local"
"Louisiana","+13185089128","User not assigned; Local"
"Daviess County - I...","+18125419365","User not assigned; Local"
"Carroll county, In...","+15746525071","User not assigned; Local"
"gwinnett county, G...","+14707065621","User not assigned; Local"
"chatham county,geo...","+19124999268","User not assigned; Local"
"Small Counties in ...","+14077076143","User not assigned; Local"
"citrus county, flo...","+13524991258","User not assigned; Local"
"Small Counties in ...","+17705258809","User not assigned; Local"
"cobb county, georg...","+16789292216","User not assigned; Local"
"225 area code","+12254248103","User not assigned; Local"
"Kansas","+13163955183","User not assigned; Local"
"Alaska's number ne...","+17254254404","User not assigned; Local"
"Harrison, Mississi...","+12284004036","User not assigned; Local"
"Muscogee, Georgia","+19183038573","User not assigned; Local"
"Mississippi Master...","+16622225934","User not assigned; Local"
"Alaska's number mo...","+14068023404","User not assigned; Local"
"Middlesex County, ...","+17326466236","User not assigned; Local"
"Maryland Mastershe...","+13012731905","User not assigned; Local"
"Bergen County, New...","+12015617650","User not assigned; Local"
"Hamilton County, N...","+13156303846","User not assigned; Local"
"Walton County, FL","+14482311813","User not assigned; Local"
"Kansas Mastersheet...","+16202702775","User not assigned; Local"
"Rockingham, New Ha...","+18025007954","User not assigned; Local"
"Nebraska Mastershe...","+14022512697","User not assigned; Local"
"Alaska's number or...","+19719838355","User not assigned; Local"
"essex county, nj-","+19735429513","User not assigned; Local"
"Alaska's number mi...","+13142073793","User not assigned; Local"
"Frederick County, ...","+12404543396","User not assigned; Local"
"Mississippi Master...","+16013688041","Hazem Hamza; Local"
"Clay county, Misso...","+18164398731","User not assigned; Local"
"washoe county, nev...","+17754383486","User not assigned; Local"
"Pasco County","+17274728014","User not assigned; Local"
"lowndes county, ge...","+12293542878","User not assigned; Local"
"South Carolina","+18036803618","User not assigned; Local"
"camden county, new...","+18562307034","User not assigned; Local"
"kent county, delaw...","+13022005986","User not assigned; Local"
"Guilford County, N...","+17439027840","User not assigned; Local"
"minnesota mastersh...","+15075790782","User not assigned; Local"
"missouri mastershe...","+15734554247","User not assigned; Local"
"Mercer County, New...","+16097397749","User not assigned; Local"
"Wake County, North...","+19842408215","User not assigned; Local"
"Utah master sheet","+13855265144","User not assigned; Local"
"Coos County, New H...","+12074814053","Mike Mo; Local"
"Williamson County,...","+16159949770","User not assigned; Local"
"Buncome County, nc","+18289444485","User not assigned; Local"
"Pennsylvania Maste...","+18149626651","User not assigned; Local"
"Anderson County, T...","+18653539979","User not assigned; Local"
"Ohio Mastersheet","+16149572308","User not assigned; Local"
"Cocke County, Tn","+14234829877","User not assigned; Local"
"Coffee county, Tn","+19312818593","User not assigned; Local"
"Mecklenburg County...","+19803507750","User not assigned; Local"
"Northdakota","+17018470849","User not assigned; Local"
"Alaska south dakot...","+16052105579","User not assigned; Local"
"Texas Mastersheet ...","+12109341449","User not assigned; Local"
"fairfax county, vi...","+17032916415","User not assigned; Local"
"Onslow County, Nor...","+12523802361","User not assigned; Local"
"Collin County, TX","+12143909687","User not assigned; Local"
"Osage County-","+15392170307","User not assigned; Local"
"Virginia","+12765660919","User not assigned; Local"
"OHIO","+13305425538","User not assigned; Local"
"Wyoming","+13073066031","User not assigned; Local"
"Henrico County, Vi...","+18043737742","User not assigned; Local"
"Greene County, Ohi...","+19379091859","User not assigned; Local"
"Central Alberta Re...","+18254454207","User not assigned; Local"
"Augusta County, Vi...","+15404100889","User not assigned; Local"
"Hays County,Texas","+17372588222","User not assigned; Local"
"Dewey County, OK,","+15804224142","User not assigned; Local"
"henderson county, ...","+17312064877","User not assigned; Local"
"Milwaukee County,W...","+14143125720","User not assigned; Local"
"Texas Master Sheet...","+19724337634","User not assigned; Local"
"Mariam's number 58","+18172867537","User not assigned; Local"
"DANE COUNTY, WI","+16086846788","User not assigned; Local"
"guadalupe county, ...","+18303964084","User not assigned; Local"
"Alaska british col...","+12362404182","Alaska .; Local"
"Luka's number 2","+447888864285","User not assigned; Mobile"
"Mercer County, New...","+17248604063","Luka Kvrivishvili; Local"
"Saskatoon","+16393879172","User not assigned; Local"
"Somerset County, N...","+19085168503","User not assigned; Local"
"Nueces County, Tex...","+13612400099","User not assigned; Local"
"Michigan Mastershe...","+12484536638","Mariam .; Local"
"Wisconsin","+12624060854","Rowayna .; Local"
"HAMILTON - NIAGRA","+12893029448","User not assigned; Local"
"Shams's number 5","+18592176955","User not assigned; Local"
"Montgomery County,...","+19362899134","User not assigned; Local"
"+19312832920","+19312832920","User not assigned"
"Johnston County 2","",""
"""

def load_default_pool() -> list[dict]:
    df = pd.read_csv(
        io.StringIO(DEFAULT_CSV),
        dtype=str,
        keep_default_na=False,   # keep "" instead of NaN
    )
    needed = {"label", "phone"}
    if not needed.issubset(df.columns):
        raise ValueError("Default CSV must include 'label' and 'phone' columns.")
    return df.to_dict("records")

# ----------------------------- Timezone Table -----------------------------

TZ_GROUPS = {
    "USA": [
        ("Central (CST/CDT)", "America/Chicago"),
        ("Mountain (MST/MDT)", "America/Denver"),
        ("Pacific (PST/PDT)", "America/Los_Angeles"),
        ("Alaska (AKST/AKDT)", "America/Anchorage"),
        ("Hawaii-Aleutian (HST)", "Pacific/Honolulu"),
    ],
    "Canada": [
        ("Newfoundland (NST/NDT)", "America/St_Johns"),
        ("Atlantic (AST/ADT)", "America/Halifax"),
        ("Eastern (EST/EDT)", "America/Toronto"),
        ("Central (CST/CDT)", "America/Winnipeg"),
        ("Mountain (MST/MDT)", "America/Edmonton"),
        ("Pacific (PST/PDT)", "America/Vancouver"),
    ],
    "UK": [
        ("United Kingdom (GMT/BST)", "Europe/London"),
    ],
    "Australia": [
        ("Western (AWST)", "Australia/Perth"),
        ("Northern Territory (ACST, no DST)", "Australia/Darwin"),
        ("Queensland (AEST, no DST)", "Australia/Brisbane"),
        ("New South Wales/VIC/TAS/ACT (AEDT/AEST)", "Australia/Sydney"),
        ("South Australia (ACDT/ACST)", "Australia/Adelaide"),
    ],
}

def build_time_table(est_hours: List[int], group_key: str) -> pd.DataFrame:
    est_tz = ZoneInfo("America/New_York")
    cols = ["EST"] + [label for (label, _) in TZ_GROUPS[group_key]]
    rows = []
    today_est = datetime.now(est_tz).date()
    for h in est_hours:
        est_dt = datetime(today_est.year, today_est.month, today_est.day, h, 0, tzinfo=est_tz)
        row = [est_dt.strftime("%-I:%M %p")]
        for label, tzname in TZ_GROUPS[group_key]:
            loc_dt = est_dt.astimezone(ZoneInfo(tzname))
            row.append(loc_dt.strftime("%-I:%M %p"))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

# ----------------------------- Streamlit Web UI -----------------------------

st.set_page_config(page_title="Cold Call Number Selector", page_icon="📞", layout="centered")
st.title("📞 Cold Call Number Selector")
st.write("Upload your phone number pool or use the default. Then enter a county/state to find the top 3 most local numbers.")

uploaded_file = st.file_uploader("Upload your CSV (must have at least 'phone' and 'label' columns). Extra columns like 'notes' are kept.", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    pool = df.to_dict('records')
    st.success(f"Loaded {len(pool)} numbers from uploaded CSV.")
else:
    pool = load_default_pool()
    st.info(f"Using your default pool of {len(pool)} numbers.")

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

from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

# Initialize timezone and geolocation tools
geolocator = Nominatim(user_agent="cold_call_timezone")
tf = TimezoneFinder()

# Helper function to build EST time range
def _time_range_to_hours(est_start, est_end, step_minutes):
    est_tz = ZoneInfo("America/New_York")
    today = datetime.now(est_tz).date()
    start_dt = datetime.combine(today, est_start, tzinfo=est_tz)
    end_dt = datetime.combine(today, est_end, tzinfo=est_tz)
    times = []
    cur = start_dt
    while cur <= end_dt:
        times.append(cur)
        cur += timedelta(minutes=int(step_minutes))
    return times

# ----------------------------- Timezone Comparison (EST anchor) -----------------------------
st.header("🕑 Timezone comparison (EST anchor)")

with st.expander("Show timezone tables", expanded=True):

    # --- City input and detection ---
    location_input = st.text_input("Enter any city or state (US, Canada, UK, Australia):", value="New York")
    detected_tzname = None
    detected_label = None

    try:
        loc = geolocator.geocode(location_input)
        if loc:
            tzname = tf.timezone_at(lat=loc.latitude, lng=loc.longitude)
            if tzname:
                detected_tzname = tzname
                # Try to find which known group this timezone belongs to
                for region, zones in TZ_GROUPS.items():
                    for label, tz in zones:
                        if tz == tzname:
                            detected_label = label
                            break
                if not detected_label:
                    detected_label = tzname.split("/")[-1].replace("_", " ")
                st.success(f"**{location_input.title()}** is in timezone: `{tzname}` ({detected_label})")
            else:
                st.error("Could not determine timezone for that location.")
        else:
            st.error("Could not find that city or state. Please try again.")
    except Exception as e:
        st.error(f"Error: {e}")

    # --- Full timezone comparison table (always visible) ---
    est_start = st.time_input("EST start time", value=pd.to_datetime("09:00").time())
    est_end = st.time_input("EST end time", value=pd.to_datetime("16:00").time())
    step_minutes = st.number_input("Step (minutes)", value=60, min_value=15, max_value=180, step=15)

    times = _time_range_to_hours(est_start, est_end, step_minutes)

    # Collect all unique timezones and labels
    tz_columns = [("Eastern (EST/EDT)", "America/New_York")]
    for region in TZ_GROUPS.values():
        for label, tzname in region:
            if tzname not in [t for _, t in tz_columns]:
                tz_columns.append((label, tzname))

    # Build table rows
    table_rows = []
    for est_dt in times:
        row = [est_dt.strftime("%-I:%M %p")]
        for _, tzname in tz_columns[1:]:  # skip the first (EST) since it's already anchor
            loc_dt = est_dt.astimezone(ZoneInfo(tzname))
            row.append(loc_dt.strftime("%-I:%M %p"))
        table_rows.append(row)

    # Build dataframe (columns line up perfectly now)
    col_names = [label for label, _ in tz_columns]
    tz_df = pd.DataFrame(table_rows, columns=col_names)
    st.dataframe(tz_df, use_container_width=True)
    st.caption("Tip: The full timezone table is always visible. Enter a city above to detect its timezone automatically.")

st.caption("Developed for NSR Cold Calling Agent – Streamlit edition.")

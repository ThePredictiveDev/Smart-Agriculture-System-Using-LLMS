# Smart Agriculture Analytics – Robust Demo v5
# -----------------------------------------------------------
# New tweaks (per user request):
#   • Re‑added **fertilization_status** to prediction keys.
#   • Convert wttr.in temps (°F) → °C for display & prompt; align sensor temp ±5 °C.
#   • Units appended in tables (°C, %, kg/ha, etc.).
#   • Display names prettified (underscores → spaces, title‑case).
#   • Explicit section headers inserted above each output card via Gradio Markdown.
#
# -------- Imports & Config --------
import os, json, random, asyncio, re, time, logging
from typing import Dict, Any
import requests, requests_cache, jsonschema
import gradio as gr

# Set up logging for key events
logging.basicConfig(filename="key_events.log", level=logging.INFO, format="%(asctime)s - %(message)s")

GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN", "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")
MODEL = "deepseek-r1-distill-llama-70b"
requests_cache.install_cache("http_cache", expire_after=900, allowable_methods=("GET"))

# -------- Prediction Keys --------
ALL_KEYS = [
    "soil_type", "possible_pests", "crop_recommendation", 
    "harvest_time", "selling_time", "possible_real_time_threats", "irrigation_recommendation",
    "fertilizer_recommendation", "fertilization_status"
]
SOIL_KEYS = ["soil_type", "fertilizer_recommendation", "fertilization_status"]
CROP_KEYS = ["crop_recommendation", "harvest_time", "selling_time"]

# -------- Hardcoded Sensor Values --------
# These will be populated with actual temperature and humidity on startup
# Rest of the values are fixed for each key
HARDCODED_SENSORS = {
    "k": {"N": 40, "P": 30, "K": 20, "soil_moisture": 55.0},
    "l": {"N": 40, "P": 30, "K": 20, "soil_moisture": 65.0},
    "t": {"N": 50, "P": 60, "K": 50, "soil_moisture": 30.0},
    "y": {"N": 50, "P": 60, "K": 50, "soil_moisture": 40.0},
    "m": {"N": 10, "P": 15, "K": 12, "soil_moisture": 75.0},
    "n": {"N": 10, "P": 15, "K": 12, "soil_moisture": 85.0},
    "s": {"N": 45, "P": 40, "K": 35, "soil_moisture": 40.0},
    "d": {"N": 45, "P": 40, "K": 35, "soil_moisture": 50.0}
}

# Variable to store the last key pressed by any user
LAST_KEY_PRESSED = None

# Base weather temperature - will be populated on first fetch
BASE_WEATHER_TEMP = None
BASE_WEATHER_HUMIDITY = None

# -------- Helpers --------
WTTR = "https://wttr.in/Mandi?format=j1"

def f_to_c(f):
    try:
        return round((float(f) - 32) * 5 / 9, 1)
    except Exception:
        return None

def fetch_weather():
    global BASE_WEATHER_TEMP, BASE_WEATHER_HUMIDITY
    try:
        r = requests.get(WTTR, timeout=5)
        d = r.json() if r.status_code == 200 else {}
    except Exception:
        return {"current": {}, "forecast": []}
    cur = d.get("current_condition", [{}])[0]
    current = {
        "temperature": f_to_c(cur.get("temp_F")),
        "wind_speed": cur.get("windspeedMiles"),
        "humidity": cur.get("humidity"),
        "condition": cur.get("weatherDesc", [{"value": "N/A"}])[0]["value"],
    }
    
    # Store base weather temp and humidity for hardcoded sensors if not already set
    if BASE_WEATHER_TEMP is None and current.get("temperature") is not None:
        BASE_WEATHER_TEMP = current.get("temperature")
    if BASE_WEATHER_HUMIDITY is None and current.get("humidity") is not None:
        BASE_WEATHER_HUMIDITY = int(current.get("humidity"))
        
        # Initialize hardcoded sensor values with weather-based temperature and humidity
        for key in HARDCODED_SENSORS:
            HARDCODED_SENSORS[key]["temperature"] = round(BASE_WEATHER_TEMP - random.uniform(0, 2), 1)
            HARDCODED_SENSORS[key]["humidity"] = round(float(BASE_WEATHER_HUMIDITY) - random.uniform(0, 5), 1)
    
    forecast = []
    for f in d.get("weather", []):
        h0 = f.get("hourly", [{}])[0]
        forecast.append({
            "date": f.get("date"),
            "condition": h0.get("weatherDesc", [{"value": "N/A"}])[0]["value"],
            "temperature": f_to_c(f.get("avgtempF")),
            "wind_speed": h0.get("windspeedMiles"),
            "humidity": h0.get("humidity"),
        })
    return {"current": current, "forecast": forecast}

def log_key_event(key: str):
    """Log the key event from any user."""
    global LAST_KEY_PRESSED
    if key and key.lower() in HARDCODED_SENSORS:
        LAST_KEY_PRESSED = key.lower()
        logging.info(f"Key pressed: {key}")
        return f"Key logged: {key}"
    return ""

def gen_sensor(weather_cur):
    # If a key has been pressed, use the hardcoded values
    global LAST_KEY_PRESSED
    if LAST_KEY_PRESSED in HARDCODED_SENSORS:
        sensor_vals = HARDCODED_SENSORS[LAST_KEY_PRESSED].copy()
        # Reset the key so it's only used once
        LAST_KEY_PRESSED = None
        return sensor_vals
    
    # Otherwise, generate random values as before
    base_temp = weather_cur.get("temperature")
    temp = round(random.uniform(base_temp - 5, base_temp + 5), 1) if base_temp else round(random.uniform(15, 40), 1)
    base_hum = int(weather_cur.get("humidity"))
    humidity = round(random.uniform(base_hum - 10, base_hum + 10), 1) if base_hum else round(random.uniform(30, 70), 1)
    return {
        "temperature": temp,  # °C
        "humidity": humidity,  # %
        "N": random.randint(0, 100),  # ppm (assumed)
        "P": random.randint(0, 100),  # ppm
        "K": random.randint(0, 100),  # ppm
        "soil_moisture": round(random.uniform(0, 100), 1),  # %
    }

# -------- Prompt Builder --------
SCHEMA = {
    "type": "object",
    "properties": {
        "sensor_data": {"type": "object"},
        "weather_data": {"type": "object"},
        "predictions": {"type": "object"},
    },
    "required": ["sensor_data", "weather_data", "predictions"],
}
SCHEMA_STR = json.dumps(SCHEMA, indent=2)
location = "Mandi, Himachal Pradesh, India"

BASE_INSTR = (
    f"You are an agronomy analytics model currently based in {location}. Respond with exactly one JSON object matching this schema:\n"
    + SCHEMA_STR
    + "\nRules:\n"
    "• Do NOT add commentary. Output JSON only.\n"
    "• Field semantics:\n"
    "  - soil_type → Only predict one of these soil types: 'Loamy' (default, most common), 'Clayey', or 'Red'. Base this on NPK values. Most fields should be 'Loamy'.\n"
    "  - possible_pests → comma‑separated likely pests given conditions (None if unlikely).\n"
    f"  - crop_recommendation → crop recommendation based on current date, weather, location: {location} and soil conditions.\n"
    f"  - harvest_time → optimal harvest date considering {location} climate & current date. Use format like 'Mid March 2023', 'Late September 2023', etc.\n"
    f"  - selling_time → optimal selling date using seasonality for {location} markets. Use format like 'Early April 2023', 'Late October 2023', etc.\n"
    "  - fertilization_status → Over, Under, or Optimal based on Nitrogen: N, Phosphorus: P, Pottasium: K values. If any one is low judge based on your understanding, if two or more are low then under. Same goes for values being high\n"
    "  - fertilizer_recommendation → Give in N:P:K form (N P K Ratio of fertilizer) based on current N,P,K values and common commercially available fertilizer N:P:K ratios if fertilization status is optimal or over then just output None.\n"
    "  - possible_real_time_threats → comma‑separated threats likely now (weather, disease). Focus on common agricultural threats like 'Heavy rainfall', 'Drought', 'Frost', 'Powdery mildew', etc. If none, output 'None'.\n"
    "  - irrigation_recommendation → Provide specific guidance on irrigation based on soil moisture and weather. Use terms like 'Normal irrigation', 'Increase irrigation frequency', 'Reduce irrigation', etc. Always include specific suggestion.\n"
    "  - anomaly_detection → true (never output yes) if any sensor/weather metric is outside typical agronomic range and write extremely concisely what it is. Else output false (never output no)\n"
    f"\n• For most fields in {location}, prefer common crops based on conditions, climate, soil type, location and seasonality\n"
    "• For N:P:K values: Normal range is N(40-80), P(20-60), K(20-40). Below these is 'low', above is 'high'.\n"
    "• For moisture: Below 30% is dry, 30-60% is normal, above 60% is wet.\n"
    "• For temperature: Below 15°C is cold, 15-35°C is normal, above 35°C is hot.\n"
    "• For humidity: Below 30% is dry, 30-70% is normal, above 70% is humid.\n"
)

FENCE_RE = re.compile(r"```json|```", re.I)
HEADERS = {"Authorization": f"Bearer {GROQ_API_TOKEN}", "Content-Type": "application/json"}


def build_prompt(sensor, weather, keys, adv=""):
    body = (
        f"Sensor Data (°C, %, ppm):\n{json.dumps(sensor, indent=2)}\n\n"
        f"Weather Data (°C):\n{json.dumps(weather, indent=2)}\n\n"
        f"Predict ONLY these keys in 'predictions': {', '.join(keys)}."
    )
    if adv:
        body += f"\nAdditional instructions: {adv}"
    return BASE_INSTR + "\n\n" + body

# -------- LLM Wrapper --------
async def call_llm(prompt):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "top_p": 0.5, "seed": random.randint(1, 1000000)}
    for _ in range(3):
        try:
            r = requests.post(GROQ_API_URL, json=payload, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        await asyncio.sleep(3)
    return {"error": "LLM request failed"}

# -------- JSON Extract --------

def extract_json(text):
    txt = FENCE_RE.sub("", text).strip()
    start = txt.find("{")
    bal = 0
    for i, ch in enumerate(txt[start:], start):
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
            if bal == 0:
                try:
                    return json.loads(txt[start : i + 1])
                except Exception:
                    return None
    return None

# -------- Formatters --------

def pretty_name(k):
    return k.replace("_", " ").title()

UNIT_MAP = {"temperature": "°C", "humidity": "%", "soil_moisture": "%", "wind_speed": "mph"}

def table(data):
    rows = []
    for k, v in data.items():
        unit = " " + UNIT_MAP.get(k, "") if UNIT_MAP.get(k) else ""
        rows.append(f"<tr><td>{pretty_name(k)}</td><td><b>{v}{unit}</b></td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def sensor_card(s):
    return table(s)

def weather_card(w):
    cur = table(w.get("current", {}))
    fore = "".join([f"<li>{f['date']}: {f['condition']} ({f['temperature']}°C)</li>" for f in w.get("forecast", [])])
    return cur + "<br><ul>" + fore + "</ul>"

def preds_md(preds):
    # Build the basic text content as before
    md = ""
    for k, v in preds.items():
        if isinstance(v, bool):
            v = "Yes" if v else "No"
        md += f"* **{pretty_name(k)}:** {v}\n"
    
    # Add raw JSON in details section
    md += "\n<details><summary>Raw JSON</summary><pre>" + json.dumps(preds, indent=2) + "</pre></details>"
    return md

# Helper functions for visualization
def extract_status_from_prediction(preds_dict, keywords, default):
    # Extract status from prediction dictionary based on keywords
    status = default
    for key, value in preds_dict.items():
        if any(keyword in key.lower() for keyword in keywords):
            # Extract words that might indicate status
            for word in ['low', 'high', 'medium', 'normal', 'healthy', 'poor', 'good', 
                         'excellent', 'deficient', 'excessive', 'adequate', 'inadequate']:
                if word in value.lower():
                    status = word
                    break
    return status

def get_status_color(status, is_risk=False):
    # Return color based on status
    status = status.lower()
    
    if is_risk:
        # For risk factors (like pests) - reversed logic
        if status in ['low', 'minimal']:
            return '#4CAF50'  # Green - good
        elif status in ['medium', 'moderate']:
            return '#FF9800'  # Orange - warning
        elif status in ['high', 'severe']:
            return '#F44336'  # Red - danger
        else:
            return '#9E9E9E'  # Grey - unknown
    else:
        # For positive factors (like health)
        if status in ['good', 'high', 'healthy', 'excellent', 'normal', 'adequate']:
            return '#4CAF50'  # Green - good
        elif status in ['medium', 'moderate', 'fair']:
            return '#FF9800'  # Orange - warning
        elif status in ['low', 'poor', 'deficient', 'inadequate', 'unhealthy']:
            return '#F44336'  # Red - danger
        else:
            return '#9E9E9E'  # Grey - unknown

# -------- Enhanced Visualization Functions --------
def create_trend_chart(sensor, weather):
    # Create animated trend charts for environmental data with better labels
    
    # Sample data for visualization (normally would use historical data)
    soil_data = [
        int(sensor.get('soil_moisture', 50)) - 10, 
        int(sensor.get('soil_moisture', 50)) - 5, 
        int(sensor.get('soil_moisture', 50)) - 2, 
        int(sensor.get('soil_moisture', 50))
    ]
    
    temp_data = [
        int(float(sensor.get('temperature', 25))) - 3, 
        int(float(sensor.get('temperature', 25))) - 2, 
        int(float(sensor.get('temperature', 25))) - 1, 
        int(float(sensor.get('temperature', 25)))
    ]
    
    humidity_data = [
        int(sensor.get('humidity', 60)) - 8, 
        int(sensor.get('humidity', 60)) - 4, 
        int(sensor.get('humidity', 60)) - 2, 
        int(sensor.get('humidity', 60))
    ]
    
    html = f"""
    <div style="margin-bottom:20px;">
        <h4 style="margin-bottom:10px; display:flex; align-items:center;">
            <span style="margin-right:8px;">Environmental Trends</span>
            <span style="font-size:12px; font-weight:normal; background:#f0f0f0; padding:2px 8px; border-radius:4px;">24-hour data</span>
        </h4>
        <div style="display:flex; gap:15px; overflow-x:auto;">
            <!-- Soil Moisture Chart -->
            <div style="min-width:220px; background:#f9f9f9; border-radius:8px; padding:15px; border:1px solid #eee;">
                <h5 style="margin:0 0 10px 0; font-size:14px; color:#689F38;">
                    <span style="display:inline-block; width:12px; height:12px; background:#8BC34A; border-radius:50%; margin-right:6px;"></span>
                    Soil Moisture
                </h5>
                <div style="height:100px; display:flex; align-items:flex-end; gap:8px; margin-bottom:5px;">
                    {create_bar(soil_data[0], soil_data[-1], 1, '#8BC34A')}
                    {create_bar(soil_data[1], soil_data[-1], 2, '#8BC34A')}
                    {create_bar(soil_data[2], soil_data[-1], 3, '#8BC34A')}
                    {create_bar(soil_data[3], soil_data[-1], 4, '#8BC34A')}
                </div>
                <div style="text-align:right; font-size:16px; font-weight:bold; color:#689F38;">
                    {soil_data[-1]}%
                </div>
                <div style="text-align:center; font-size:11px; color:#666; margin-top:5px;">
                    Higher is better
                </div>
            </div>
            
            <!-- Temperature Chart -->
            <div style="min-width:220px; background:#f9f9f9; border-radius:8px; padding:15px; border:1px solid #eee;">
                <h5 style="margin:0 0 10px 0; font-size:14px; color:#E64A19;">
                    <span style="display:inline-block; width:12px; height:12px; background:#FF5722; border-radius:50%; margin-right:6px;"></span>
                    Temperature
                </h5>
                <div style="height:100px; display:flex; align-items:flex-end; gap:8px; margin-bottom:5px;">
                    {create_bar(temp_data[0], temp_data[-1], 1, '#FF5722')}
                    {create_bar(temp_data[1], temp_data[-1], 2, '#FF5722')}
                    {create_bar(temp_data[2], temp_data[-1], 3, '#FF5722')}
                    {create_bar(temp_data[3], temp_data[-1], 4, '#FF5722')}
                </div>
                <div style="text-align:right; font-size:16px; font-weight:bold; color:#E64A19;">
                    {temp_data[-1]}°C
                </div>
                <div style="text-align:center; font-size:11px; color:#666; margin-top:5px;">
                    Current field temperature
                </div>
            </div>
            
            <!-- Humidity Chart -->
            <div style="min-width:220px; background:#f9f9f9; border-radius:8px; padding:15px; border:1px solid #eee;">
                <h5 style="margin:0 0 10px 0; font-size:14px; color:#0288D1;">
                    <span style="display:inline-block; width:12px; height:12px; background:#03A9F4; border-radius:50%; margin-right:6px;"></span>
                    Humidity
                </h5>
                <div style="height:100px; display:flex; align-items:flex-end; gap:8px; margin-bottom:5px;">
                    {create_bar(humidity_data[0], humidity_data[-1], 1, '#03A9F4')}
                    {create_bar(humidity_data[1], humidity_data[-1], 2, '#03A9F4')}
                    {create_bar(humidity_data[2], humidity_data[-1], 3, '#03A9F4')}
                    {create_bar(humidity_data[3], humidity_data[-1], 4, '#03A9F4')}
                </div>
                <div style="text-align:right; font-size:16px; font-weight:bold; color:#0288D1;">
                    {humidity_data[-1]}%
                </div>
                <div style="text-align:center; font-size:11px; color:#666; margin-top:5px;">
                    Ambient air humidity
                </div>
            </div>
        </div>
    </div>
    """
    return html

def create_bar(value, max_value, delay, color):
    # Helper function to create animated bars for charts
    max_value = max(max_value, value, 1)  # Avoid division by zero
    percent = min(int((value / max_value) * 100), 100)
    return f"""
    <div style="flex:1; position:relative;">
        <div style="position:absolute; bottom:0; left:0; right:0; background:{color}; 
                    height:0%; animation:grow-{delay} 1.5s forwards; opacity:0.7;"></div>
        <style>
        @keyframes grow-{delay} {{
            0% {{ height: 0%; }}
            100% {{ height: {percent}%; }}
        }}
        </style>
    </div>
    """

def create_status_badges(preds_dict):
    # Create status badges with pulsing indicators and improved descriptions
    
    # Extract indicators from predictions
    nutrient_level = extract_status_from_prediction(preds_dict, ['soil', 'nutrient', 'fertilization'], 'normal')
    water_status = extract_status_from_prediction(preds_dict, ['irrigation', 'water'], 'adequate')
    plant_health = extract_status_from_prediction(preds_dict, ['crop', 'plant'], 'healthy')
    pest_risk = extract_status_from_prediction(preds_dict, ['pest', 'disease', 'threat'], 'low')
    
    # Get direct values from predictions
    fertilization_status = preds_dict.get('fertilization_status', nutrient_level).title()
    crop_rec = preds_dict.get('crop_recommendation', 'Not available')
    possible_pests = preds_dict.get('possible_pests', 'None detected')
    threats = preds_dict.get('possible_real_time_threats', 'None detected')
    soil_type = preds_dict.get('soil_type', 'Loamy')
    fertilizer_rec = preds_dict.get('fertilizer_recommendation', 'None')
    
    # Use actual prediction values if available
    if fertilization_status and fertilization_status.lower() != 'none':
        nutrient_level = fertilization_status
    
    if threats and threats.lower() != 'none':
        pest_risk = 'High' if 'danger' in threats.lower() or 'severe' in threats.lower() else 'Moderate'
    
    if possible_pests and possible_pests.lower() != 'none':
        if ',' in possible_pests:
            pest_risk = 'High'
        else:
            pest_risk = 'Moderate'
    
    html = f"""
    <div style="margin-bottom:20px;">
        <h4 style="margin-bottom:10px;">Farm Status Summary</h4>
        <div style="display:flex; flex-wrap:wrap; gap:10px;">
            {create_badge("Nutrient Level", nutrient_level, get_status_color(nutrient_level), 
                          "NPK balance in soil", f"Fertilizer Recommendation: {fertilizer_rec}")}
            
            {create_badge("Water Status", water_status, get_status_color(water_status), 
                          "Field irrigation needs", preds_dict.get('irrigation_recommendation', 'Monitor soil moisture'))}
            
            {create_badge("Plant Health", plant_health, get_status_color(plant_health), 
                          f"Soil Type: {soil_type}", f"Crop Recommendation: {crop_rec}")}
            
            {create_badge("Pest Risk", pest_risk, get_status_color(pest_risk, True), 
                          "Threat assessment", f"Pests: {possible_pests if possible_pests and possible_pests.lower() != 'none' else 'None'}")}
        </div>
    </div>
    """
    return html

def create_badge(label, status, color, subtitle, tooltip=""):
    # Create a badge with pulsing indicator and improved information
    return f"""
    <div style="background:#f9f9f9; border-radius:6px; padding:12px; border:1px solid #eee; min-width:120px; width:calc(50% - 25px);">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <div style="width:12px; height:12px; border-radius:50%; background:{color}; position:relative;">
                <div style="position:absolute; top:-3px; left:-3px; right:-3px; bottom:-3px; 
                            border-radius:50%; border:2px solid {color}; opacity:0.5;
                            animation:pulse-small 2s infinite;"></div>
            </div>
            <div style="font-size:13px; font-weight:bold;">{label}</div>
        </div>
        <div style="font-size:15px; text-transform:capitalize; color:{color}; font-weight:bold;">
            {status}
        </div>
        <div style="font-size:11px; color:#666; margin-top:5px;">
            {subtitle}
            <div style="margin-top:4px; font-style:italic; font-size:10px;">{tooltip}</div>
        </div>
    </div>
    
    <style>
    @keyframes pulse-small {{
        0% {{ transform:scale(1); opacity:0.5; }}
        70% {{ transform:scale(1.5); opacity:0; }}
        100% {{ transform:scale(1); opacity:0; }}
    }}
    </style>
    """

def create_ai_recommendation(preds_dict):
    # Display the actual model predictions in a structured format
    
    # Extract key recommendations from prediction data
    recommendations = []
    for key, value in preds_dict.items():
        if key == "possible_real_time_threats" and value and value.lower() != "none":
            recommendations.append(f"<strong>ALERT!</strong> Detected threats: {value}")
        
        if key == "irrigation_recommendation" and value:
            recommendations.append(f"<strong>Irrigation:</strong> {value}")
        
        if key == "fertilizer_recommendation" and value and value != "None":
            recommendations.append(f"<strong>Fertilizer (N:P:K):</strong> {value}")
        
        if key == "crop_recommendation" and value:
            recommendations.append(f"<strong>Crop recommendation:</strong> {value}")
        
        if key == "soil_type" and value:
            recommendations.append(f"<strong>Soil type:</strong> {value}")
        
        if key == "harvest_time" and value:
            recommendations.append(f"<strong>Optimal harvest:</strong> {value}")
        
        if key == "selling_time" and value:
            recommendations.append(f"<strong>Best selling time:</strong> {value}")
    
    # If no specific recommendations were found, show general advice
    if not recommendations:
        recommendations = [
            "<strong>Maintain current practices:</strong> Continue regular monitoring",
            "<strong>Weather awareness:</strong> Monitor local weather forecast",
            "<strong>Soil testing:</strong> Consider periodic soil testing for optimal management"
        ]
    
    # Join all recommendations with line breaks
    rec_text = "<br>".join(recommendations)
    
    html = f"""
    <div style="margin-bottom:10px;">
        <h4 style="margin-bottom:10px; display:flex; align-items:center;">
            <span style="margin-right:8px;">AI Recommendations</span>
            <span style="font-size:12px; font-weight:normal; color:green; display:flex; align-items:center;">
                <span style="display:inline-block; width:8px; height:8px; background:green; border-radius:50%; margin-right:4px;"></span>
                LLM-powered analysis
            </span>
        </h4>
        <div style="background:#f0f7ff; border:1px solid #cce5ff; border-radius:8px; padding:15px; 
                    position:relative; overflow:hidden;">
            <div style="position:absolute; top:0; right:0; width:120px; height:120px; background:radial-gradient(circle, rgba(0,123,255,0.1) 0%, rgba(0,123,255,0) 70%);"></div>
            
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="background:rgba(0,123,255,0.1); width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-right:10px;">
                    <span style="font-size:18px;">🧠</span>
                </div>
                <div>
                    <div style="font-weight:bold; font-size:15px;">Smart Agriculture Analysis</div>
                    <div style="font-size:12px; opacity:0.7;">Based on sensor data and weather patterns</div>
                </div>
            </div>
            
            <div id="recommendation-text" class="typing-text" style="line-height:1.5;">
            </div>
            
            <!-- Predictions summary -->
            <div style="margin-top:15px; background:rgba(255,255,255,0.7); padding:10px; border-radius:5px; font-size:13px;">
                <div style="font-weight:bold; margin-bottom:5px;">All Predictions:</div>
                <div style="max-height:150px; overflow-y:auto;">
                    {format_predictions_html(preds_dict)}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        (function() {{
            const text = `{rec_text}`;
            const element = document.getElementById('recommendation-text');
            let i = 0;
            
            function typeWriter() {{
                if (i < text.length) {{
                    element.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 30);
                }}
            }}
            
            // Start the typing effect when element is visible
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        typeWriter();
                        observer.disconnect();
                    }}
                }});
            }});
            
            observer.observe(element);
        }})();
    </script>
    
    <style>
    .typing-text::after {{
        content: '|';
        animation: blink 1s infinite;
    }}
    
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0; }}
    }}
    </style>
    """
    return html

def format_predictions_html(preds_dict):
    """Format all predictions as HTML for display"""
    if not preds_dict:
        return "<em>No predictions available</em>"
    
    items = []
    for key, value in preds_dict.items():
        # Skip items already prominently displayed
        if key in ["irrigation_recommendation", "fertilizer_recommendation", 
                  "possible_real_time_threats", "crop_recommendation"]:
            continue
        
        # Format the key name nicely
        nice_key = key.replace("_", " ").title()
        
        # Add the item
        items.append(f"<div><strong>{nice_key}:</strong> {value}</div>")
    
    return "".join(items)

def create_weather_display(weather):
    # Extract weather data
    current = weather.get("current", {})
    forecast = weather.get("forecast", [])
    
    # Get current weather info
    current_temp = current.get("temperature", "N/A")
    current_humidity = current.get("humidity", "N/A")
    current_condition = current.get("condition", "N/A")
    current_wind = current.get("wind_speed", "N/A")
    
    # Create weather icon based on condition
    weather_icon = "☀️"  # default sunny
    if current_condition:
        condition_lower = current_condition.lower()
        if "rain" in condition_lower or "shower" in condition_lower:
            weather_icon = "🌧️"
        elif "cloud" in condition_lower or "overcast" in condition_lower:
            weather_icon = "☁️"
        elif "snow" in condition_lower:
            weather_icon = "❄️"
        elif "fog" in condition_lower or "mist" in condition_lower:
            weather_icon = "🌫️"
        elif "thunder" in condition_lower or "storm" in condition_lower:
            weather_icon = "⛈️"
        elif "clear" in condition_lower or "sunny" in condition_lower:
            weather_icon = "☀️"
        elif "partly" in condition_lower:
            weather_icon = "⛅"
    
    # Format forecast data
    forecast_html = ""
    if forecast:
        for i, day in enumerate(forecast[:3]):  # Show up to 3 days forecast
            date = day.get("date", "")
            condition = day.get("condition", "N/A")
            temp = day.get("temperature", "N/A")
            
            # Create forecast icon
            forecast_icon = "☀️"  # default
            if condition:
                condition_lower = condition.lower()
                if "rain" in condition_lower or "shower" in condition_lower:
                    forecast_icon = "🌧️"
                elif "cloud" in condition_lower or "overcast" in condition_lower:
                    forecast_icon = "☁️"
                elif "snow" in condition_lower:
                    forecast_icon = "❄️"
                elif "fog" in condition_lower or "mist" in condition_lower:
                    forecast_icon = "🌫️"
                elif "thunder" in condition_lower or "storm" in condition_lower:
                    forecast_icon = "⛈️"
                elif "clear" in condition_lower or "sunny" in condition_lower:
                    forecast_icon = "☀️"
                elif "partly" in condition_lower:
                    forecast_icon = "⛅"
            
            forecast_html += f"""
            <div style="flex:1; min-width:100px; background:rgba(255,255,255,0.7); padding:8px; border-radius:5px; text-align:center;">
                <div style="font-weight:bold; font-size:12px;">{date}</div>
                <div style="font-size:20px; margin:5px 0;">{forecast_icon}</div>
                <div style="font-size:14px;">{condition}</div>
                <div style="font-weight:bold; margin-top:3px;">{temp}°C</div>
            </div>
            """
    
    html = f"""
    <div style="margin-bottom:20px;">
        <h4 style="margin-bottom:10px; display:flex; align-items:center;">
            <span style="margin-right:8px;">Weather - Mandi, Himachal Pradesh</span>
            <span style="font-size:12px; font-weight:normal; color:#666; display:flex; align-items:center;">
                <span style="display:inline-block; width:8px; height:8px; background:#03A9F4; border-radius:50%; margin-right:4px;"></span>
                Live data from wttr.in
            </span>
        </h4>
        
        <div style="display:flex; gap:15px; flex-wrap:wrap;">
            <!-- Current Weather -->
            <div style="flex:1; min-width:220px; background:linear-gradient(135deg, #4b6cb7, #182848); border-radius:8px; padding:15px; color:white; position:relative; overflow:hidden;">
                <div style="position:absolute; top:0; right:0; font-size:64px; opacity:0.2; line-height:1; margin-right:10px;">{weather_icon}</div>
                
                <div style="font-size:16px; margin-bottom:5px;">Current Conditions</div>
                <div style="font-size:28px; font-weight:bold; margin-bottom:15px;">{current_temp}°C</div>
                
                <div style="display:flex; gap:15px; margin-top:10px;">
                    <div>
                        <div style="font-size:13px; opacity:0.8;">Condition</div>
                        <div style="font-weight:bold;">{current_condition}</div>
                    </div>
                    <div>
                        <div style="font-size:13px; opacity:0.8;">Humidity</div>
                        <div style="font-weight:bold;">{current_humidity}%</div>
                    </div>
                    <div>
                        <div style="font-size:13px; opacity:0.8;">Wind</div>
                        <div style="font-weight:bold;">{current_wind} mph</div>
                    </div>
                </div>
            </div>
            
            <!-- Forecast -->
            <div style="flex:2; min-width:260px; background:#f5f5f5; border-radius:8px; padding:15px; border:1px solid #eee;">
                <div style="font-weight:bold; margin-bottom:10px; font-size:14px;">3-Day Forecast</div>
                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    {forecast_html}
                </div>
            </div>
        </div>
    </div>
    """
    return html

# -------- Enhanced Results Display --------
def enhanced_output(sensor, weather, preds):
    # Create enhanced visualization with map, chart and badges
    try:
        # Parse predictions if they're in string format
        if isinstance(preds, str):
            # Extract just prediction content
            preds_dict = {}
            for line in preds.split('\n'):
                if ':' in line and '*' in line:
                    key = line.split('**')[1].replace(':', '').strip()
                    value = line.split(':')[1].strip()
                    preds_dict[key.lower().replace(' ', '_')] = value
        else:
            preds_dict = preds
            
        # Create the enhanced HTML
        html = f"""
        <div style="font-family:system-ui,-apple-system,'Segoe UI',Roboto,Ubuntu,'Open Sans',sans-serif;">
            <h3 style="margin-bottom:15px;">Smart Farm Analytics Dashboard</h3>
            {create_weather_display(weather)}
            {create_trend_chart(sensor, weather)}
            {create_status_badges(preds_dict)}
            {create_ai_recommendation(preds_dict)}
        </div>
        """
        return html
    except Exception as e:
        # Fallback to basic display if error
        print(f"Error in enhanced output: {e}")
        return f"""
        <div style="padding:20px; background:#f8d7da; border-radius:8px; color:#721c24; margin-top:20px;">
            <h3>Dashboard Error</h3>
            <p>Unable to create enhanced view. Please check the Raw Data tab for information.</p>
            <details>
                <summary>Error Details</summary>
                <pre style="background:#f5f5f5; padding:10px; border-radius:5px;">{str(e)}</pre>
            </details>
        </div>
        """

# -------- Inference --------
async def inference(mode, specific, set_list, adv):
    weather = fetch_weather()
    sensor = gen_sensor(weather.get("current", {}))
        
    if mode == "All Predictions":
        keys = ALL_KEYS
    elif mode == "Soil Related Predictions":
        keys = SOIL_KEYS
    elif mode == "Crop Related Predictions":
        keys = CROP_KEYS
    elif mode == "Specific Prediction":
        keys = [specific]
    else:
        keys = set_list
    
    prompt = build_prompt(sensor, weather, keys, adv)
    resp = await call_llm(prompt)
    
    if "error" in resp: 
        error_msg = "**Error:** " + resp["error"]
        return sensor_card(sensor), weather_card(weather), error_msg, f"""
        <div style="padding:20px; background:#f8d7da; border-radius:8px; color:#721c24; margin-top:20px;">
            <h3>Error</h3>
            <p>{resp["error"]}</p>
        </div>
        """
    
    content = resp["choices"][0]["message"]["content"]
    obj = extract_json(content)
    
    if not obj: 
        return sensor_card(sensor), weather_card(weather), "Model output unparsable.", """
        <div style="padding:20px; background:#f8d7da; border-radius:8px; color:#721c24; margin-top:20px;">
            <h3>Error</h3>
            <p>Could not parse model output. Please try again.</p>
        </div>
        """
    
    # Get the enhanced visualization
    enhanced_html = enhanced_output(sensor, weather, obj.get("predictions", {}))
    
    # Return both basic data cards and enhanced visualization
    return sensor_card(sensor), weather_card(weather), preds_md(obj.get("predictions", {})), enhanced_html

# -------- UI --------
CSS = """
body { color: #222 !important; }
.card { background:#fff; border:1px solid #ccc; border-radius:8px; padding:12px; }
.card, .card * { color:#222 !important; }
.gr-markdown, .gr-markdown * { color:#222 !important; }
.gr-html, .gr-html * { color:#222 !important; }
details summary { cursor:pointer; }
.enhanced { margin-top:10px; }

/* Ensure Smart Dashboard tab content is visible */
.tab-content { min-height:300px; }

/* Hide the key input box but keep it functional */
#key-input { position: absolute; opacity: 0; pointer-events: none; }

/* Minimal styling for the discrete sensor input */
.discrete-input { opacity: 0.7; }
.discrete-input:hover { opacity: 1; }
.discrete-input input { color: white !important; }
.discrete-input input::placeholder { color: white !important; opacity: 0.5; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌾 Smart Agriculture Analytics – Demo v5")
    
    # Hidden textbox to receive key events from JavaScript - using an ID for easier targeting
    keybox = gr.Textbox(value="", label="Key Input", elem_id="key-input", interactive=True)
    
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(label="Query Mode", choices=["All Predictions","Soil Related Predictions","Crop Related Predictions","Specific Prediction","Set of Predictions"], value="All Predictions")
            specific = gr.Dropdown(label="Specific Prediction", visible=False, choices=ALL_KEYS)
            setbox = gr.CheckboxGroup(label="Set of Predictions", visible=False, choices=ALL_KEYS)
            adv = gr.Textbox(label="Advanced Prompt (optional)")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            # Tabs for basic and enhanced views
            with gr.Tabs(selected=0) as tabs:
                with gr.TabItem("Smart Dashboard", id="dashboard_tab"):
                    enhanced_html = gr.HTML(elem_classes=["enhanced", "tab-content"])
                with gr.TabItem("Raw Data", id="raw_data_tab"):
                    gr.Markdown("### Sensor Snapshot")
                    sensor_html = gr.HTML(elem_classes="card")
                    gr.Markdown("### Weather Snapshot")
                    weather_html = gr.HTML(elem_classes="card")
                    gr.Markdown("### Predictions")
                    preds_markdown = gr.Markdown(elem_classes="card")
    
    # Add a minimal, discrete input at the bottom
    with gr.Row():
        # Empty column for spacing
        with gr.Column(scale=3):
            gr.HTML("&nbsp;")
        
        # Minimal sensor input
        with gr.Column(scale=1, elem_classes=["discrete-input"]):
            manual_key = gr.Textbox(
                label="",
                placeholder="",
                max_lines=1,
                interactive=True,
                container=False,
                scale=1
            )
            manual_submit = gr.Button("→", size="sm")
            
        # Empty column for spacing
        with gr.Column(scale=3):
            gr.HTML("&nbsp;")

    # Improved JavaScript to capture key presses
    js_code = """
    <script>
    // Wait for the page to fully load
    document.addEventListener('DOMContentLoaded', function() {
        // Function to handle key presses
        function handleKeyPress(event) {
            const validKeys = ['k','l','t','y','m','n','s','d'];
            const key = event.key.toLowerCase();
            
            if (validKeys.includes(key)) {
                // Find the input element by ID
                const keyInput = document.getElementById('key-input');
                if (keyInput) {
                    // Set value and manually trigger events
                    keyInput.value = key;
                    
                    // Create and dispatch events
                    keyInput.dispatchEvent(new Event('input', { bubbles: true }));
                    keyInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }
        
        // Add event listener to the entire document
        document.addEventListener('keydown', handleKeyPress);
        
        // Also add to window for good measure
        window.addEventListener('keydown', handleKeyPress);
    });
    </script>
    """
    gr.HTML(js_code)

    def toggle(m):
        if m == "Specific Prediction":
            return gr.update(visible=True), gr.update(visible=False)
        if m == "Set of Predictions":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)
    
    # Function to process manual key input - simplified with no feedback
    def process_manual_key(key):
        if key and len(key) >= 1:
            first_char = key[0].lower()
            if first_char in HARDCODED_SENSORS:
                log_key_event(first_char)
            return ""  # Clear the input field
        return key

    mode.change(toggle, mode, [specific, setbox])
    btn.click(inference, [mode, specific, setbox, adv], [sensor_html, weather_html, preds_markdown, enhanced_html])
    
    # Connect the key event handler - no visible feedback
    keybox.change(log_key_event, keybox, None)
    
    # Connect the manual key input - no visible feedback
    manual_submit.click(process_manual_key, manual_key, manual_key)

if __name__ == "__main__":
    demo.launch(share=True)

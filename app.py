import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
import json
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated")

# ================== إعدادات الصفحة ==================
st.set_page_config(
    page_title="نظام مساندة القرار للزراعة الذكية",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS مخصص ==================
st.markdown("""
<style>
    .main {background-color: #f0f8f5;}
    .stAlert {border-radius: 10px;}
    h1 {color: #2d6a4f; text-align: center;}
    .case-study-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-right: 5px solid #4CAF50;
    }
    .cost-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================== توليد البيانات التاريخية ==================
@st.cache_data
def generate_historical_data(n_samples=500):
    """توليد بيانات تاريخية واقعية للتدريب"""
    np.random.seed(42)
    
    data = {
        'temperature': np.random.normal(25, 7, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        'rainfall': np.random.exponential(5, n_samples),
        'soil_moisture': np.random.normal(45, 10, n_samples),
        'ph_level': np.random.normal(6.5, 0.8, n_samples),
        'nitrogen': np.random.normal(40, 10, n_samples),
        'phosphorus': np.random.normal(35, 8, n_samples),
        'potassium': np.random.normal(30, 7, n_samples),
        'crop_type': np.random.choice(['طماطم', 'خيار', 'قمح', 'ذرة', 'خس'], n_samples),
        'soil_type': np.random.choice(['طينية', 'رملية', 'صفراء'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['yield'] = (
        (df['temperature'].clip(15, 35) / 35) * 30 +
        (df['humidity'].clip(30, 80) / 80) * 25 +
        (df['soil_moisture'].clip(20, 70) / 70) * 25 +
        (df['nitrogen'].clip(20, 60) / 60) * 10 +
        (df['phosphorus'].clip(20, 50) / 50) * 5 +
        (df['potassium'].clip(15, 45) / 45) * 5 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    df['water_need'] = (
        df['temperature'] * 0.5 +
        (100 - df['humidity']) * 0.3 +
        df['soil_moisture'].apply(lambda x: 30 if x < 30 else 20 if x < 50 else 15) +
        np.random.normal(0, 3, n_samples)
    ).clip(10, 50)
    
    return df

# ================== بناء نموذج AI ==================
@st.cache_resource
def train_ml_models():
    """تدريب نماذج التعلم الآلي"""
    df = generate_historical_data(500)
    
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
    
    features = ['temperature', 'humidity', 'rainfall', 'soil_moisture', 
                'ph_level', 'nitrogen', 'phosphorus', 'potassium', 
                'crop_encoded', 'soil_encoded']
    
    X = df[features]
    y_yield = df['yield']
    y_water = df['water_need']
    
    model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
    model_yield.fit(X, y_yield)
    
    model_water = RandomForestRegressor(n_estimators=100, random_state=42)
    model_water.fit(X, y_water)
    
    return model_yield, model_water, le_crop, le_soil

# ================== بيانات احتياطية ==================
def get_fallback_weather():
    """بيانات احتياطية في حالة فشل الـ API"""
    return {
        'temperature': 25.0,
        'humidity': 60.0,
        'rainfall': 0.0,
        'wind_speed': 10.0,
        'description': 'معتدل',
        'pressure': 1013.0,
        'visibility': 10.0
    }

# ================== API الطقس الحقيقي ==================
def get_real_weather(city="Cairo", api_key=None):
    """جلب بيانات الطقس الحقيقية من OpenWeatherMap"""
    
    # التحقق من وجود API key
    if not api_key or api_key.strip() == "":
        st.warning("⚠️ لم يتم إدخال API Key - يتم استخدام بيانات وهمية")
        weather_data = {
            'temperature': np.random.normal(25, 5),
            'humidity': np.random.normal(60, 10),
            'rainfall': max(0, np.random.exponential(3) if np.random.random() > 0.7 else 0),
            'wind_speed': np.random.uniform(5, 25),
            'description': np.random.choice(['صافي', 'غائم جزئياً', 'ممطر', 'مشمس']),
            'pressure': np.random.uniform(1010, 1020),
            'visibility': np.random.uniform(8, 10)
        }
        return weather_data
    
    try:
        # URL للـ API
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        # المعاملات
        params = {
            'q': city,
            'appid': api_key.strip(),
            'units': 'metric',  # لدرجة الحرارة بالسيليزيوس
            'lang': 'ar'        # الوصف بالعربي
        }
        
        # إرسال الطلب
        response = requests.get(base_url, params=params, timeout=10)
        
        # فحص الاستجابة
        if response.status_code == 200:
            data = response.json()
            
            # استخراج البيانات
            weather_data = {
                'temperature': float(data['main']['temp']),
                'humidity': float(data['main']['humidity']),
                'rainfall': float(data.get('rain', {}).get('1h', 0)),  # المطر في آخر ساعة
                'wind_speed': float(data['wind']['speed']) * 3.6,  # تحويل من m/s إلى km/h
                'description': data['weather'][0]['description'] if data.get('weather') else 'غير متاح',
                'pressure': float(data['main']['pressure']),
                'visibility': float(data.get('visibility', 10000)) / 1000  # تحويل لـ km
            }
            
            st.success(f"✅ تم جلب بيانات الطقس الحقيقية من {city} بنجاح!")
            return weather_data
            
        elif response.status_code == 401:
            st.error("❌ API Key غير صحيح! تحقق من المفتاح.")
            st.info("💡 احصل على مفتاح مجاني من: https://openweathermap.org/api")
            return get_fallback_weather()
            
        elif response.status_code == 404:
            st.error(f"❌ المدينة '{city}' غير موجودة! جرب اسم مدينة آخر بالإنجليزية.")
            return get_fallback_weather()
            
        else:
            st.error(f"❌ خطأ في الاتصال بالـ API: {response.status_code}")
            return get_fallback_weather()
            
    except requests.exceptions.Timeout:
        st.error("❌ انتهى وقت الاتصال - تحقق من الإنترنت")
        return get_fallback_weather()
        
    except requests.exceptions.ConnectionError:
        st.error("❌ خطأ في الاتصال بالإنترنت")
        return get_fallback_weather()
        
    except Exception as e:
        st.error(f"❌ خطأ غير متوقع: {str(e)}")
        return get_fallback_weather()

# ================== توليد بيانات الطقس ==================
def generate_weather_forecast(days=7):
    """توليد توقعات الطقس"""
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    base_temp = 25
    temps = [base_temp + np.random.normal(0, 5) + np.sin(i/7*2*np.pi)*3 for i in range(days)]
    humidity = [60 + np.random.normal(0, 10) - i*2 for i in range(days)]
    rainfall = [max(0, np.random.exponential(3) if np.random.random() > 0.6 else 0) for _ in range(days)]
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'temperature': np.round(temps, 1),
        'humidity': np.round(np.clip(humidity, 30, 90), 1),
        'rainfall': np.round(rainfall, 1)
    })
    
    return forecast_df

# ================== معلومات المحاصيل ==================
CROPS_INFO = {
    'طماطم': {
        'icon': '🍅', 
        'growth_days': 80, 
        'min_temp': 18, 
        'max_temp': 30, 
        'ideal_ph': 6.5,
        'cost_per_kg': 2.5,
        'price_per_kg': 5.0,
        'yield_per_m2': 8
    },
    'خيار': {
        'icon': '🥒', 
        'growth_days': 60, 
        'min_temp': 20, 
        'max_temp': 32, 
        'ideal_ph': 6.0,
        'cost_per_kg': 2.0,
        'price_per_kg': 4.5,
        'yield_per_m2': 10
    },
    'قمح': {
        'icon': '🌾', 
        'growth_days': 120, 
        'min_temp': 15, 
        'max_temp': 25, 
        'ideal_ph': 6.5,
        'cost_per_kg': 1.5,
        'price_per_kg': 3.0,
        'yield_per_m2': 5
    },
    'ذرة': {
        'icon': '🌽', 
        'growth_days': 90, 
        'min_temp': 18, 
        'max_temp': 35, 
        'ideal_ph': 6.0,
        'cost_per_kg': 1.8,
        'price_per_kg': 3.5,
        'yield_per_m2': 6
    },
    'خس': {
        'icon': '🥬', 
        'growth_days': 45, 
        'min_temp': 12, 
        'max_temp': 20, 
        'ideal_ph': 6.5,
        'cost_per_kg': 3.0,
        'price_per_kg': 6.0,
        'yield_per_m2': 4
    }
}

SOIL_INFO = {
    'طينية': {'retention': 0.8, 'drainage': 0.3, 'nutrients': 0.9},
    'رملية': {'retention': 0.3, 'drainage': 0.9, 'nutrients': 0.4},
    'صفراء': {'retention': 0.6, 'drainage': 0.6, 'nutrients': 0.7}
}

# ================== دراسات الحالة ==================
CASE_STUDIES = {
    'حالة 1: مزرعة طماطم صغيرة': {
        'crop': 'طماطم',
        'soil': 'طينية',
        'area': 500,
        'soil_moisture': 55,
        'ph': 6.5,
        'nitrogen': 45,
        'phosphorus': 38,
        'potassium': 32,
        'water': 2000,
        'description': 'مزرعة صغيرة في منطقة معتدلة المناخ، تربة خصبة، موارد مياه جيدة'
    },
    'حالة 2: مشروع خيار تجاري': {
        'crop': 'خيار',
        'soil': 'صفراء',
        'area': 1000,
        'soil_moisture': 48,
        'ph': 6.2,
        'nitrogen': 42,
        'phosphorus': 35,
        'potassium': 28,
        'water': 3500,
        'description': 'مشروع تجاري متوسط، تربة متوازنة، هدف تحقيق أعلى إنتاجية'
    },
    'حالة 3: مزرعة قمح في بيئة صعبة': {
        'crop': 'قمح',
        'soil': 'رملية',
        'area': 2000,
        'soil_moisture': 35,
        'ph': 7.0,
        'nitrogen': 30,
        'phosphorus': 25,
        'potassium': 22,
        'water': 2500,
        'description': 'مزرعة كبيرة في بيئة صحراوية، تحدي نقص المياه والعناصر الغذائية'
    }
}

# ================== حساب التكاليف ==================
def calculate_costs(crop, area, predicted_yield, predicted_water):
    """حساب التكاليف والأرباح"""
    crop_info = CROPS_INFO[crop]
    
    seeds_cost = area * 0.5
    water_cost = predicted_water * 7 * (crop_info['growth_days'] / 7) * 0.02
    fertilizer_cost = area * 2
    labor_cost = area * 1.5
    other_costs = area * 0.8
    
    total_cost = seeds_cost + water_cost + fertilizer_cost + labor_cost + other_costs
    
    expected_yield_kg = area * crop_info['yield_per_m2'] * (predicted_yield / 100)
    revenue = expected_yield_kg * crop_info['price_per_kg']
    
    profit = revenue - total_cost
    roi = (profit / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'seeds_cost': seeds_cost,
        'water_cost': water_cost,
        'fertilizer_cost': fertilizer_cost,
        'labor_cost': labor_cost,
        'other_costs': other_costs,
        'total_cost': total_cost,
        'expected_yield_kg': expected_yield_kg,
        'revenue': revenue,
        'profit': profit,
        'roi': roi
    }

# ================== تحميل النماذج ==================
model_yield, model_water, le_crop, le_soil = train_ml_models()

# ================== الواجهة الرئيسية ==================
st.markdown("<h1>🌾 نظام مساندة القرار للزراعة الذكية - AI Powered</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>قرارات ذكية مدعومة بالذكاء الاصطناعي والبيانات الحقيقية</p>", unsafe_allow_html=True)

# ================== Tabs ==================
tab1, tab2, tab3 = st.tabs(["🏠 الرئيسية", "📊 دراسات الحالة", "💰 التحليل المالي"])

# ================== TAB 1: الرئيسية ==================
with tab1:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ إعدادات المشروع")
        
        # قسم API الطقس
        st.subheader("🌍 إعدادات الطقس")
        api_key = st.text_input(
            "OpenWeatherMap API Key",
            type="password",
            placeholder="أدخل المفتاح هنا...",
            help="احصل على مفتاح مجاني من openweathermap.org/api"
        )
        
        if not api_key:
            st.info("💡 **للحصول على API Key مجاني:**\n1. سجل في openweathermap.org\n2. اذهب لـ API Keys\n3. انسخ المفتاح والصقه هنا")
        
        city = st.text_input("🌍 المدينة (بالإنجليزية)", "Cairo", help="مثال: Riyadh, Dubai, Jeddah")
        
        if st.button("🔄 تحديث بيانات الطقس", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        # باقي الإعدادات
        selected_crop = st.selectbox(
            "🌱 اختر المحصول",
            list(CROPS_INFO.keys()),
            format_func=lambda x: f"{CROPS_INFO[x]['icon']} {x}"
        )
        
        selected_soil = st.selectbox(
            "🏜️ نوع التربة",
            list(SOIL_INFO.keys())
        )
        
        area = st.number_input("📏 المساحة (متر مربع)", 100, 10000, 500, 50)
        
        st.divider()
        st.subheader("📊 قياسات التربة")
        
        soil_moisture = st.slider("رطوبة التربة (%)", 10, 80, 45)
        ph_level = st.slider("درجة الحموضة (pH)", 4.0, 8.0, 6.5, 0.1)
        nitrogen = st.slider("نسبة النيتروجين", 10, 70, 40)
        phosphorus = st.slider("نسبة الفوسفور", 10, 60, 35)
        potassium = st.slider("نسبة البوتاسيوم", 10, 50, 30)
        
        st.divider()
        water_available = st.number_input("💧 المياه المتاحة (لتر/يوم)", 100, 5000, 1000, 50)

    # جلب بيانات الطقس
    weather_now = get_real_weather(city, api_key)
    
    # التنبؤ بالذكاء الاصطناعي
    input_features = np.array([[
        weather_now['temperature'],
        weather_now['humidity'],
        weather_now['rainfall'],
        soil_moisture,
        ph_level,
        nitrogen,
        phosphorus,
        potassium,
        le_crop.transform([selected_crop])[0],
        le_soil.transform([selected_soil])[0]
    ]])

    predicted_yield = model_yield.predict(input_features)[0]
    predicted_water = model_water.predict(input_features)[0]
    costs = calculate_costs(selected_crop, area, predicted_yield, predicted_water)

    # حالة الطقس الحالية
    st.markdown("### 🌤️ الحالة الجوية الحالية")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🌡️ الحرارة", f"{weather_now['temperature']:.1f}°C")
    with col2:
        st.metric("💧 الرطوبة", f"{weather_now['humidity']:.1f}%")
    with col3:
        st.metric("🌧️ الأمطار", f"{weather_now['rainfall']:.1f} مم")
    with col4:
        st.metric("💨 الرياح", f"{weather_now['wind_speed']:.1f} كم/س")
    with col5:
        st.metric("📊 الحالة", weather_now['description'])

    st.divider()

    # التنبؤات الذكية
    st.markdown("### 🤖 التنبؤات الذكية (AI Predictions)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0;'>{predicted_yield:.1f}%</h2>
            <p style='margin: 5px 0 0 0;'>الإنتاجية المتوقعة</p>
            <small>مقارنة بالمتوسط</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0;'>{predicted_water:.1f} لتر</h2>
            <p style='margin: 5px 0 0 0;'>احتياج المياه اليومي</p>
            <small>حسب الظروف الحالية</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        harvest_date = datetime.now() + timedelta(days=CROPS_INFO[selected_crop]['growth_days'])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;'>
            <h2 style='color: white; margin:0; font-size: 1.3em;'>{harvest_date.strftime('%d/%m/%Y')}</h2>
            <p style='margin: 5px 0 0 0;'>موعد الحصاد المتوقع</p>
            <small>{CROPS_INFO[selected_crop]['growth_days']} يوم</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # التنبيهات
    st.markdown("### ⚠️ تنبيهات ذكية")
    alerts = []
    crop_info = CROPS_INFO[selected_crop]

    if weather_now['temperature'] > crop_info['max_temp']:
        alerts.append(('warning', f"🌡️ درجة الحرارة مرتفعة جداً ({weather_now['temperature']:.1f}°C)"))
    elif weather_now['temperature'] < crop_info['min_temp']:
        alerts.append(('error', f"❄️ درجة الحرارة منخفضة جداً ({weather_now['temperature']:.1f}°C)"))

    if weather_now['rainfall'] > 10:
        alerts.append(('info', f"🌧️ أمطار غزيرة - قلل الري إلى {predicted_water*0.5:.1f} لتر"))

    if soil_moisture < 30:
        alerts.append(('warning', "💧 رطوبة التربة منخفضة - زد كمية الري"))

    if abs(ph_level - crop_info['ideal_ph']) > 1:
        alerts.append(('warning', f"⚗️ درجة حموضة التربة غير مثالية - المطلوب: {crop_info['ideal_ph']}"))

    if water_available < predicted_water * 7:
        alerts.append(('error', "🚨 كمية المياه المتاحة غير كافية للأسبوع القادم"))

    if predicted_yield < 50:
        alerts.append(('error', "📉 الظروف غير مناسبة - ننصح بتأجيل الزراعة"))

    if alerts:
        for alert_type, message in alerts:
            if alert_type == 'error':
                st.error(message)
            elif alert_type == 'warning':
                st.warning(message)
            else:
                st.info(message)
    else:
        st.success("✅ جميع الظروف مثالية للزراعة!")

    st.divider()

    # الرسوم البيانية
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 توقعات الطقس (7 أيام)")
        forecast = generate_weather_forecast(7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['temperature'],
                                mode='lines+markers', name='درجة الحرارة',
                                line=dict(color='#ff6b6b', width=3)))
        fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['humidity'],
                                mode='lines+markers', name='الرطوبة',
                                line=dict(color='#4ecdc4', width=3)))
        
        fig.update_layout(height=300, xaxis_title="التاريخ", yaxis_title="القيمة",
                         hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 تحليل العوامل المؤثرة")
        
        factors = pd.DataFrame({
            'العامل': ['درجة الحرارة', 'الرطوبة', 'العناصر الغذائية', 'رطوبة التربة', 'نوع التربة'],
            'التأثير': [
                min(100, (weather_now['temperature'] / crop_info['max_temp']) * 100),
                min(100, weather_now['humidity']),
                min(100, (nitrogen + phosphorus + potassium) / 3 * 1.2),
                soil_moisture,
                SOIL_INFO[selected_soil]['retention'] * 100
            ]
        })
        
        fig = go.Figure(go.Bar(x=factors['التأثير'], y=factors['العامل'], orientation='h',
                              marker=dict(color=factors['التأثير'], colorscale='Viridis', showscale=True)))
        fig.update_layout(height=300, xaxis_title="نسبة الملاءمة (%)", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # جدول الري
    st.markdown("### 💧 جدول الري الأسبوعي (مُحسّن بالـ AI)")

    weekly_schedule = []
    for i in range(7):
        day_temp = forecast.iloc[i]['temperature']
        day_rain = forecast.iloc[i]['rainfall']
        
        adjusted_water = predicted_water * (1 + (day_temp - 25) / 50)
        if day_rain > 5:
            adjusted_water *= 0.5
        
        morning = adjusted_water * 0.6
        evening = adjusted_water * 0.4
        
        weekly_schedule.append({
            'اليوم': forecast.iloc[i]['date'].strftime('%A'),
            'التاريخ': forecast.iloc[i]['date'].strftime('%d/%m'),
            'الصباح (لتر)': f"{morning:.1f}",
            'المساء (لتر)': f"{evening:.1f}",
            'التسميد': '✅' if i % 3 == 0 else '—',
            'ملاحظات': '🌧️ أمطار' if day_rain > 5 else '☀️ جاف'
        })

    schedule_df = pd.DataFrame(weekly_schedule)
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    total_weekly = sum([float(s['الصباح (لتر)']) + float(s['المساء (لتر)']) for s in weekly_schedule])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 إجمالي استهلاك الأسبوع", f"{total_weekly:.1f} لتر")
    with col2:
        efficiency = (1 - SOIL_INFO[selected_soil]['drainage']) * 100
        st.metric("💚 كفاءة استخدام المياه", f"{efficiency:.0f}%")
    with col3:
        savings = (water_available * 7 - total_weekly) / (water_available * 7) * 100 if water_available * 7 > 0 else 0
        st.metric("💰 التوفير المتوقع", f"{max(0, savings):.1f}%")

# ================== TAB 2: دراسات الحالة ==================
with tab2:
    st.markdown("## 📊 دراسات الحالة التطبيقية")
    st.markdown("تحليل 3 سيناريوهات واقعية مختلفة")
    
    for case_name, case_data in CASE_STUDIES.items():
        with st.expander(f"🔍 {case_name}", expanded=False):
            st.markdown(f"**الوصف:** {case_data['description']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"🌱 **المحصول:** {case_data['crop']}")
                st.write(f"🏜️ **التربة:** {case_data['soil']}")
                st.write(f"📏 **المساحة:** {case_data['area']} م²")
            with col2:
                st.write(f"💧 **رطوبة التربة:** {case_data['soil_moisture']}%")
                st.write(f"⚗️ **pH:** {case_data['ph']}")
                st.write(f"🌾 **نيتروجين:** {case_data['nitrogen']}")
            with col3:
                st.write(f"🧪 **فوسفور:** {case_data['phosphorus']}")
                st.write(f"💎 **بوتاسيوم:** {case_data['potassium']}")
                st.write(f"💧 **مياه متاحة:** {case_data['water']} لتر/يوم")
            
            # تشغيل التنبؤ لهذه الحالة
            case_input = np.array([[
                25, 60, 0,
                case_data['soil_moisture'],
                case_data['ph'],
                case_data['nitrogen'],
                case_data['phosphorus'],
                case_data['potassium'],
                le_crop.transform([case_data['crop']])[0],
                le_soil.transform([case_data['soil']])[0]
            ]])
            
            case_yield = model_yield.predict(case_input)[0]
            case_water = model_water.predict(case_input)[0]
            case_costs = calculate_costs(case_data['crop'], case_data['area'], case_yield, case_water)
            
            st.divider()
            st.markdown("### 📈 نتائج التحليل:")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("الإنتاجية", f"{case_yield:.1f}%")
            with col2:
                st.metric("الري اليومي", f"{case_water:.1f} لتر")
            with col3:
                st.metric("الربح المتوقع", f"{case_costs['profit']:.0f} ريال")
            with col4:
                st.metric("ROI", f"{case_costs['roi']:.1f}%")
            
            # رسم مقارنة
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['التكاليف', 'الإيرادات', 'الربح'],
                y=[case_costs['total_cost'], case_costs['revenue'], case_costs['profit']],
                marker_color=['#e74c3c', '#3498db', '#2ecc71'],
                text=[f"{case_costs['total_cost']:.0f}", 
                      f"{case_costs['revenue']:.0f}", 
                      f"{case_costs['profit']:.0f}"],
                textposition='auto'
            ))
            fig.update_layout(height=300, title="التحليل المالي", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

# ================== TAB 3: التحليل المالي ==================
with tab3:
    st.markdown("## 💰 التحليل المالي الشامل")
    
    # ملخص التكاليف
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💸 هيكل التكاليف")
        
        costs_data = pd.DataFrame({
            'البند': ['البذور', 'المياه', 'الأسمدة', 'العمالة', 'أخرى'],
            'التكلفة': [
                costs['seeds_cost'],
                costs['water_cost'],
                costs['fertilizer_cost'],
                costs['labor_cost'],
                costs['other_costs']
            ]
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=costs_data['البند'],
            values=costs_data['التكلفة'],
            hole=.4,
            marker_colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        )])
        fig.update_layout(height=350, title="توزيع التكاليف")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(costs_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 تحليل الربحية")
        
        profit_data = pd.DataFrame({
            'المؤشر': ['إجمالي التكاليف', 'الإيرادات المتوقعة', 'صافي الربح'],
            'القيمة (ريال)': [costs['total_cost'], costs['revenue'], costs['profit']]
        })
        
        fig = go.Figure(data=[go.Bar(
            x=profit_data['المؤشر'],
            y=profit_data['القيمة (ريال)'],
            marker_color=['#e74c3c', '#3498db', '#2ecc71'],
            text=profit_data['القيمة (ريال)'].round(2),
            textposition='auto'
        )])
        fig.update_layout(height=350, title="المؤشرات المالية", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(profit_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # المؤشرات الرئيسية
    st.markdown("### 🎯 المؤشرات الرئيسية")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='cost-card'>
            <h2 style='color: white; margin:0;'>{costs['expected_yield_kg']:.1f} كجم</h2>
            <p style='margin: 5px 0 0 0;'>الإنتاج المتوقع</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <h2 style='color: white; margin:0;'>{costs['total_cost']:.0f} ريال</h2>
            <p style='margin: 5px 0 0 0;'>إجمالي التكاليف</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <h2 style='color: white; margin:0;'>{costs['revenue']:.0f} ريال</h2>
            <p style='margin: 5px 0 0 0;'>الإيرادات</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        profit_color = '#2ecc71' if costs['profit'] > 0 else '#e74c3c'
        st.markdown(f"""
        <div class='cost-card' style='background: linear-gradient(135deg, {profit_color} 0%, {profit_color} 100%);'>
            <h2 style='color: white; margin:0;'>{costs['profit']:.0f} ريال</h2>
            <p style='margin: 5px 0 0 0;'>صافي الربح</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # مقارنة بين المحاصيل
    st.markdown("### 🌱 مقارنة الربحية بين المحاصيل")
    
    comparison_data = []
    for crop_name in CROPS_INFO.keys():
        test_input = input_features.copy()
        test_input[0][8] = le_crop.transform([crop_name])[0]
        
        test_yield = model_yield.predict(test_input)[0]
        test_costs = calculate_costs(crop_name, area, test_yield, predicted_water)
        
        comparison_data.append({
            'المحصول': f"{CROPS_INFO[crop_name]['icon']} {crop_name}",
            'الإنتاجية': test_yield,
            'التكاليف': test_costs['total_cost'],
            'الإيرادات': test_costs['revenue'],
            'الربح': test_costs['profit'],
            'ROI': test_costs['roi']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='الربح', x=comparison_df['المحصول'], 
                         y=comparison_df['الربح'], marker_color='#2ecc71'))
    fig.add_trace(go.Scatter(name='ROI %', x=comparison_df['المحصول'], 
                             y=comparison_df['ROI'], mode='lines+markers',
                             yaxis='y2', marker_color='#e74c3c', line=dict(width=3)))
    
    fig.update_layout(
        height=400,
        yaxis=dict(title='الربح (ريال)'),
        yaxis2=dict(title='ROI (%)', overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ================== Footer ==================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>نظام مساندة القرار للزراعة الذكية</strong></p>
    <p>مدعوم بالذكاء الاصطناعي | قسم نظم المعلومات الإدارية | كلية إدارة الأعمال</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        🤖 AI Models: Random Forest | 📊 Data: 500+ Records | 🌍 Real-time Weather API
    </p>
</div>
""", unsafe_allow_html=True)
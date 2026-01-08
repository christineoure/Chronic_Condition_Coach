
import json
import os
from pathlib import Path

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Sample health documents
documents = [
    {
        "source": "health_guideline",
        "title": "Sleep and Blood Sugar Regulation",
        "text": "Poor sleep reduces insulin sensitivity, leading to higher blood sugar levels. Aim for 7-8 hours of quality sleep nightly. Consistent sleep patterns help regulate cortisol and glucose metabolism. Studies show that even one night of poor sleep can increase insulin resistance by 25%."
    },
    {
        "source": "wearable_insight",
        "title": "HRV and Chronic Stress Patterns",
        "text": "Heart Rate Variability (HRV) below 40 ms often indicates chronic stress. Three consecutive days of low HRV may signal need for recovery. Morning HRV readings are most reliable for trend analysis. HRV recovery after exercise is a key marker of fitness adaptation."
    },
    {
        "source": "expert_advice",
        "title": "Diabetes Type 2 Lifestyle Triggers",
        "text": "Common triggers for blood sugar spikes: insufficient sleep, high-stress periods, excessive carbohydrate intake without fiber, and sedentary behavior after meals. Walking 10 minutes after meals can reduce postprandial glucose by 20-30%. Consistency in meal timing also helps regulate blood sugar."
    },
    {
        "source": "pattern_analysis",
        "title": "Sleep Deprivation Effects",
        "text": "Three or more nights below 6 hours of sleep correlate with: elevated resting heart rate (+5-10 bpm), increased cortisol levels, higher sugar cravings, and reduced glucose tolerance. Recovery requires 2-3 nights of quality sleep above 7 hours."
    },
    {
        "source": "coaching_principle",
        "title": "Behavior Change Strategy",
        "text": "Small, sustainable changes work better than drastic overhauls. Focus on one habit at a time. Use wearable data to validate progress and adjust strategies weekly. Celebrate small wins to maintain motivation."
    },
    {
        "source": "hypertension_guide",
        "title": "Blood Pressure Management",
        "text": "Lifestyle factors affecting hypertension: sodium intake, stress levels, sleep quality, and physical activity. Daily 30-minute moderate exercise can lower systolic BP by 4-9 mm Hg. Mindfulness practices show consistent 3-5 mm Hg reductions."
    },
    {
        "source": "asthma_insights",
        "title": "Environmental Triggers",
        "text": "Common asthma triggers include: indoor allergens, outdoor pollution, respiratory infections, and sudden weather changes. Monitoring air quality and using air purifiers can reduce exacerbations. Peak flow monitoring helps identify early warning signs."
    },
    {
        "source": "wearable_pattern",
        "title": "Step Count and Energy Levels",
        "text": "Patients with consistent step counts above 7,500 daily report 30% higher energy levels. Sudden drops in step count may precede symptom flare-ups in chronic conditions. Gradual increases of 500 steps per week are sustainable."
    },
    {
        "source": "nutrition_guidance",
        "title": "Anti-inflammatory Foods",
        "text": "Foods that reduce inflammation: fatty fish, berries, leafy greens, nuts, and olive oil. Chronic inflammation exacerbates most chronic conditions and affects pain perception. Omega-3 to Omega-6 ratio should be 1:4 or better."
    },
    {
        "source": "recovery_principles",
        "title": "Stress Recovery Techniques",
        "text": "Effective stress recovery: deep breathing (4-7-8 pattern), progressive muscle relaxation, nature exposure, and digital detox periods. Recovery is as important as activity for chronic condition management. Schedule recovery like appointments."
    },
    {
        "source": "diabetes_pattern",
        "title": "Post-Meal Glucose Control",
        "text": "Blood sugar typically peaks 60-90 minutes after meals. Light activity after eating can blunt this spike. Protein and fiber at the start of meals slow glucose absorption. Consistent carb distribution throughout the day is key."
    },
    {
        "source": "sleep_quality",
        "title": "Sleep Hygiene Practices",
        "text": "Maintain consistent sleep-wake times even on weekends. Keep bedroom cool (65-68°F) and dark. Avoid screens 1 hour before bed. Caffeine cutoff should be 8 hours before bedtime for sensitive individuals."
    }
]

# Save each document
for i, doc in enumerate(documents):
    with open(f"data/raw/doc_{i}.json", "w") as f:
        json.dump(doc, f, indent=2)

print(f" Created {len(documents)} health documents in data/raw/")
print("Documents include: sleep guidance, diabetes management, hypertension, asthma, nutrition, and wearable insights")




import streamlit as st
from PIL import Image, ImageDraw
import io

# Generer en enkel logo som SVG-tilsvarende (PNG)
def create_app_icon():
    """Generer InveStock app-ikon (48x48px for favicon)"""
    # Lager en grønn/blå finansiell chart-stil ikon
    size = 48
    img = Image.new('RGBA', (size, size), (14, 17, 23, 255))
    draw = ImageDraw.Draw(img)
    
    # Tegn en stilisert chart/graf
    # Bakgrunn
    draw.rectangle([(0, 0), (size-1, size-1)], fill=(14, 17, 23), outline=(38, 166, 154, 200), width=2)
    
    # Grøn opptrend-linje (chart)
    chart_data = [(8, 35), (15, 25), (22, 18), (29, 22), (36, 12), (42, 8)]
    draw.line(chart_data, fill=(38, 166, 154, 255), width=2)
    
    # Små prikker på linjen (data points)
    for x, y in chart_data:
        draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(38, 166, 154, 255))
    
    return img

# Generer og lagre favicon
icon = create_app_icon()
icon_bytes = io.BytesIO()
icon.save(icon_bytes, format='PNG')
icon_bytes.seek(0)

print("✅ InveStock favicon generert (48x48px)")

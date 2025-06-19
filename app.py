from flask import Flask, jsonify, send_file, render_template, request, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'your-secret-key-here'  # Needed for session management

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample initial data
land_slots = [
    {"id": 1, "x": 0, "y": 0, "status": "available", "price": "$1000", "image": None},
    {"id": 2, "x": 1, "y": 0, "status": "sold", "price": "$1500", "image": None},
    {"id": 3, "x": 0, "y": 1, "status": "pending", "price": "$1200", "image": None},
]

current_image_path = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_status_color(status):
    colors = {
        "available": "#4CAF50",  # Green
        "sold": "#F44336",       # Red
        "pending": "#FFC107",     # Amber
        "reserved": "#2196F3"     # Blue
    }
    return colors.get(status, "#9E9E9E")  # Default gray

def create_land_slots_from_image(image_path):
    """Generate land slots based on image dimensions"""
    global current_image_path
    current_image_path = image_path
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Create slots based on image size (1 slot per 200px)
            slots = []
            slot_id = 1
            for x in range(0, width, 200):
                for y in range(0, height, 200):
                    slots.append({
                        "id": slot_id,
                        "x": x // 200,
                        "y": y // 200,
                        "status": "available",
                        "price": f"${(x//200 + y//200 + 1) * 1000}",
                        "image": None
                    })
                    slot_id += 1
            return slots
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def home():
    status_filter = request.args.get('status', 'all')
    
    if status_filter == 'all':
        filtered_slots = land_slots
    else:
        filtered_slots = [slot for slot in land_slots if slot['status'] == status_filter]
    
    return render_template('index.html', 
                         land_slots=filtered_slots,
                         current_status=status_filter,
                         has_image=current_image_path is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate land slots based on image
        global land_slots
        new_slots = create_land_slots_from_image(filepath)
        if new_slots:
            land_slots = new_slots
        
        return redirect(url_for('home'))
    
    return redirect(request.url)

@app.route('/reserve/<int:slot_id>')
def reserve_slot(slot_id):
    global land_slots
    for slot in land_slots:
        if slot['id'] == slot_id and slot['status'] == 'available':
            slot['status'] = 'pending'
            break
    return redirect(url_for('home'))

@app.route('/land_slots')
def get_land_slots():
    return jsonify(land_slots)

@app.route('/generate_image')
def generate_image():
    if not current_image_path:
        return "No image uploaded", 400
    
    try:
        with Image.open(current_image_path) as base_img:
            # Create a transparent overlay
            overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                font_bold = ImageFont.truetype("arialbd.ttf", 16)
            except:
                font = ImageFont.load_default()
                font_bold = font
            
            # Draw slots on overlay
            for slot in land_slots:
                x = slot["x"] * 200 + 20
                y = slot["y"] * 200 + 20
                
                # Draw semi-transparent rectangle
                draw.rectangle([x-10, y-10, x+170, y+170], 
                             fill=get_status_color(slot['status']) + "80",  # 80 = 50% transparency
                             outline=(0, 0, 0))
                
                # Draw slot info
                draw.text((x, y), f"Plot {slot['x']},{slot['y']}", font=font_bold, fill=(0, 0, 0))
                draw.text((x, y+30), slot['status'].upper(), font=font_bold, fill=(0, 0, 0))
                draw.text((x, y+60), slot['price'], font=font, fill=(0, 0, 0))
            
            # Combine base image with overlay
            final_img = Image.alpha_composite(
                base_img.convert('RGBA'),
                overlay
            )
            
            # Save to bytes buffer in original format
            img_io = io.BytesIO()
            format = base_img.format if base_img.format else 'PNG'
            final_img.save(img_io, format)
            img_io.seek(0)
            
            return send_file(img_io, mimetype=f'image/{format.lower()}')
    
    except Exception as e:
        return f"Error generating image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify, send_file, render_template, request, redirect, url_for, flash
from PIL import Image, ImageDraw, ImageFont
import io
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch

# Load once at startup:
device = "cpu"  # or "mps" for Apple Silicon, or "cuda" for NVIDIA GPU

model_path = "ibm-granite/granite-3.3-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'your-secret-key-here'  # Needed for session management

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample initial data with enhanced structure
land_slots = [
    {"id": 1, "x": 0, "y": 0, "status": "available", "price": "$1000", "image": None, "reserved_at": None, "purchased_at": None, "buyer_info": None},
    {"id": 2, "x": 1, "y": 0, "status": "sold", "price": "$1500", "image": None, "reserved_at": "2024-01-15 10:30:00", "purchased_at": "2024-01-15 11:00:00", "buyer_info": {"name": "John Doe", "email": "john@example.com", "phone": "123-456-7890"}},
    {"id": 3, "x": 0, "y": 1, "status": "pending", "price": "$1200", "image": None, "reserved_at": "2024-01-16 09:15:00", "purchased_at": None, "buyer_info": {"name": "Jane Smith", "email": "jane@example.com", "phone": "098-765-4321"}},
]

current_image_path = None

def build_granite_prompt(budget, zone, size=None):
    land_data = "\n".join([
        f"ID {slot['id']}: {slot['price']}, zone: {slot.get('zone','N/A')}, size: {slot.get('size','N/A')}, status: {slot['status']}"
        for slot in land_slots if slot['status']=='available'
    ])
    prompt = f"""
You are an assistant helping users find suitable land plots.

User preferences:
- Budget: ${budget}
- Zone: {zone}
- Minimum size: {size or 'Any'}

Here are available plots:
{land_data}

Based on this, recommend the best options and explain why.
Only include plots within budget and correct zone.
"""
    return prompt.strip()

def query_granite_model(prompt):
    """
    Calls the local Granite 3.3 8B instruct model via Hugging Face.
    """
    output = pipe(prompt)
    # `output` is a list of dicts, we just want the generated text:
    return output[0]["generated_text"]



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
                        "image": None,
                        "reserved_at": None,
                        "purchased_at": None,
                        "buyer_info": None
                    })
                    slot_id += 1
            return slots
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def home():
    status_filter = request.args.get('status', 'all')
    search_query = request.args.get('search', '').strip()
    
    # Filter by status
    if status_filter == 'all':
        filtered_slots = land_slots
    else:
        filtered_slots = [slot for slot in land_slots if slot['status'] == status_filter]
    
    # Filter by search query
    if search_query:
        filtered_slots = [slot for slot in filtered_slots if 
                         search_query.lower() in str(slot['id']).lower() or
                         search_query.lower() in f"{slot['x']},{slot['y']}".lower() or
                         search_query.lower() in slot['price'].lower() or
                         (slot['buyer_info'] and search_query.lower() in slot['buyer_info']['name'].lower())]
    
    return render_template('index.html', 
                         land_slots=filtered_slots,
                         current_status=status_filter,
                         search_query=search_query,
                         has_image=current_image_path is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        if not file.filename:
            return redirect(request.url)
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

@app.route('/reserve/<int:slot_id>', methods=['GET', 'POST'])
def reserve_slot(slot_id):
    global land_slots
    
    if request.method == 'POST':
        # Get buyer information from form
        buyer_name = request.form.get('buyer_name')
        buyer_email = request.form.get('buyer_email')
        buyer_phone = request.form.get('buyer_phone')
        
        if not all([buyer_name, buyer_email, buyer_phone]):
            flash('Please fill in all buyer information fields.', 'error')
            return redirect(url_for('home'))
        
        # Find and update the slot
        for slot in land_slots:
            if slot['id'] == slot_id and slot['status'] == 'available':
                slot['status'] = 'pending'
                slot['reserved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                slot['buyer_info'] = {
                    'name': buyer_name,
                    'email': buyer_email,
                    'phone': buyer_phone
                }
                flash(f'Plot {slot["x"]},{slot["y"]} has been reserved for {buyer_name}.', 'success')
                break
        else:
            flash('Slot not found or not available.', 'error')
    
    return redirect(url_for('home'))

@app.route('/confirm_purchase/<int:slot_id>', methods=['POST'])
def confirm_purchase(slot_id):
    global land_slots
    
    # Get payment information
    payment_method = request.form.get('payment_method')
    payment_reference = request.form.get('payment_reference')
    
    if not payment_method or not payment_reference:
        flash('Please provide payment method and reference.', 'error')
        return redirect(url_for('home'))
    
    # Find and update the slot
    for slot in land_slots:
        if slot['id'] == slot_id and slot['status'] == 'pending':
            slot['status'] = 'sold'
            slot['purchased_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add payment information to buyer_info
            if slot['buyer_info']:
                slot['buyer_info']['payment_method'] = payment_method
                slot['buyer_info']['payment_reference'] = payment_reference
            
            flash(f'Purchase confirmed for Plot {slot["x"]},{slot["y"]}. Payment reference: {payment_reference}', 'success')
            break
    else:
        flash('Slot not found or not in pending status.', 'error')
    
    return redirect(url_for('home'))

@app.route('/cancel_reservation/<int:slot_id>')
def cancel_reservation(slot_id):
    global land_slots
    
    for slot in land_slots:
        if slot['id'] == slot_id and slot['status'] == 'pending':
            slot['status'] = 'available'
            slot['reserved_at'] = None
            slot['buyer_info'] = None
            flash(f'Reservation cancelled for Plot {slot["x"]},{slot["y"]}.', 'success')
            break
    else:
        flash('Slot not found or not in pending status.', 'error')
    
    return redirect(url_for('home'))

@app.route('/slot_details/<int:slot_id>')
def slot_details(slot_id):
    slot = None
    for s in land_slots:
        if s['id'] == slot_id:
            slot = s
            break
    
    if not slot:
        flash('Slot not found.', 'error')
        return redirect(url_for('home'))
    
    return render_template('slot_details.html', slot=slot)

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
                
                # Add buyer info for sold/pending slots
                if slot['buyer_info'] and slot['status'] in ['sold', 'pending']:
                    buyer_name = slot['buyer_info']['name']
                    draw.text((x, y+90), f"Buyer: {buyer_name[:15]}", font=font, fill=(0, 0, 0))
            
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
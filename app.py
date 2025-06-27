from flask import Flask, jsonify, send_file, render_template, request, redirect, url_for, flash, session
from markupsafe import Markup
from PIL import Image, ImageDraw, ImageFont
import io
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import re

# AI Model Configuration
device = "cpu"  # Force CPU usage
model_path = "ibm-granite/granite-3.3-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'your-secret-key-here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample initial data with enhanced structure
land_slots = [
    {"id": 1, "x": 0, "y": 0, "status": "available", "price": "$1000", "image": None, "reserved_at": None, "purchased_at": None, "buyer_info": None},
    {"id": 2, "x": 1, "y": 0, "status": "sold", "price": "$1500", "image": None, "reserved_at": "2024-01-15 10:30:00", "purchased_at": "2024-01-15 11:00:00", "buyer_info": {"name": "John Doe", "email": "john@example.com", "phone": "123-456-7890"}},
    {"id": 3, "x": 0, "y": 1, "status": "pending", "price": "$1200", "image": None, "reserved_at": "2024-01-16 09:15:00", "purchased_at": None, "buyer_info": {"name": "Jane Smith", "email": "jane@example.com", "phone": "098-765-4321"}},
]

current_image_path = None

# AI Helper Functions
def build_granite_prompt(budget, zone, size=None):
    """Constructs a prompt for the AI model with available land data"""
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
    """Queries the local Granite model for recommendations"""
    output = pipe(prompt)
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
    page = int(request.args.get('page', 1))
    per_page = 9
    
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

    # Filter by recommended IDs (if present)
    recommended = request.args.get('recommended')
    if recommended:
        ids = [int(x) for x in recommended.split(',')]
        filtered_slots = [slot for slot in filtered_slots if slot['id'] in ids]
        # Optionally, set a flag to show a message like "Showing recommended plots"

    total_slots = len(filtered_slots)
    total_pages = (total_slots + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_slots = filtered_slots[start:end]
    
    # Summary statistics for all slots 
    all_slots = land_slots
    total_count = len(all_slots)
    available_count = sum(1 for slot in all_slots if slot['status'] == 'available')
    pending_count = sum(1 for slot in all_slots if slot['status'] == 'pending')
    sold_count = sum(1 for slot in all_slots if slot['status'] == 'sold')
    reserved_count = sum(1 for slot in all_slots if slot['status'] == 'reserved')

    return render_template('index.html', 
                         land_slots=paginated_slots,
                         current_status=status_filter,
                         search_query=search_query,
                         has_image=current_image_path is not None,
                         page=page,
                         total_pages=total_pages,
                         total_count=total_count,
                         available_count=available_count,
                         pending_count=pending_count,
                         sold_count=sold_count,
                         reserved_count=reserved_count)

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
            
            img_io = io.BytesIO()
            format = base_img.format if base_img.format else 'PNG'
            final_img.save(img_io, format)
            img_io.seek(0)
            
            return send_file(img_io, mimetype=f'image/{format.lower()}')
    
    except Exception as e:
        return f"Error generating image: {str(e)}", 500

@app.route('/ai_recommend', methods=['GET', 'POST'])
def ai_recommend():
    if request.method == 'POST':
        budget = request.form.get('budget')
        zone = request.form.get('zone')
        size = request.form.get('size')
        
        if not budget or not zone:
            flash('Please provide at least budget and zone information.', 'error')
            return redirect(url_for('ai_recommend'))
        
        try:
            prompt = build_granite_prompt(budget, zone, size)
            recommendation = query_granite_model(prompt)
            recommendation = remove_prompt_echo(prompt, recommendation)
            recommended_ids = extract_ids_from_recommendation(recommendation)
            if recommended_ids:
                return redirect(url_for('home', recommended=','.join(map(str, recommended_ids))))
        except Exception as e:
            flash(f'Error generating recommendation: {str(e)}', 'error')
            return redirect(url_for('ai_recommend'))
    
    return render_template('ai_recommendation.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_form():
    if request.method == 'POST':
        budget = request.form.get('budget')
        zone = request.form.get('zone')
        size = request.form.get('size')

        if not budget or not zone:
            flash('Please provide at least budget and zone information.', 'error')
            return redirect(url_for('recommend_form'))

        try:
            prompt = build_granite_prompt(budget, zone, size)
            recommendation = query_granite_model(prompt)
            recommendation = remove_prompt_echo(prompt, recommendation)
            recommended_ids = extract_ids_from_recommendation(recommendation)
            if recommended_ids:
                return redirect(url_for('home', recommended=','.join(map(str, recommended_ids))))
        except Exception as e:
            flash(f'Error generating recommendation: {str(e)}', 'error')
            return redirect(url_for('recommend_form'))

    return render_template('recommend_form.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        # Save user info to session
        session['user'] = {
            'avatar': request.form.get('avatar', ''),
            'name': request.form.get('name', ''),
            'email': request.form.get('email', ''),
        }
        return redirect(url_for('profile'))

    user = session.get('user', {
        'avatar': '',
        'name': '',
        'email': '',
    })
    # Calculate user's reserved and sold plots
    reserved_count = 0
    sold_count = 0
    for slot in land_slots:
        if slot.get('buyer_info') and user.get('name') and slot['buyer_info'].get('name') == user['name']:
            if slot['status'] == 'pending':
                reserved_count += 1
            elif slot['status'] == 'sold':
                sold_count += 1
    user['reserved_count'] = reserved_count
    user['sold_count'] = sold_count
    
    user['owned_count'] = reserved_count + sold_count
    user['pending_count'] = reserved_count
    return render_template('profile.html', user=user)

@app.route('/map')
def map_view():
    return render_template('map.html', land_slots=land_slots)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        chat_history = session['chat_history']

        if user_message:
            # Add user message to history
            chat_history.append({'role': 'user', 'content': user_message})

            # Build prompt with land slot data and chat history
            land_data = "\n".join([
                f"ID {slot['id']}: {slot['price']}, zone: {slot.get('zone','N/A')}, size: {slot.get('size','N/A')}, status: {slot['status']}"
                for slot in land_slots
            ])
            history_text = "\n".join(
                [f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in chat_history]
            )
            prompt = f"""
You are a helpful real estate assistant. Here is the current land slot data:
{land_data}

Conversation so far:
{history_text}

Based on the above, answer the user's last question or provide recommendations if asked.
""".strip()

            try:
                ai_response = query_granite_model(prompt)
                ai_response = remove_prompt_echo(prompt, ai_response)
            except Exception as e:
                ai_response = f"Sorry, I couldn't process your request: {str(e)}"

            chat_history.append({'role': 'assistant', 'content': ai_response})
            session['chat_history'] = chat_history

        return render_template('chatbot.html', chat_history=chat_history)

    return render_template('chatbot.html', chat_history=session.get('chat_history', []))

def remove_prompt_echo(prompt, response):
    response_clean = response.lstrip()
    prompt_clean = prompt.strip()
    if response_clean.startswith(prompt_clean):
        return response_clean[len(prompt_clean):].lstrip()
    for sep in ["\n\n", "\n", ":"]:
        if sep in response_clean:
            parts = response_clean.split(sep, 1)
            if len(parts) > 1:
                return parts[1].lstrip()
    return response_clean 

def extract_ids_from_recommendation(recommendation):
    ids = re.findall(r'ID\s*(\d+)|Plot\s*(\d+)', recommendation)
    flat_ids = [int(num) for tup in ids for num in tup if num]
    if not flat_ids:
        flat_ids = [int(x) for x in re.findall(r'\b\d+\b', recommendation)]
    return sorted(set(flat_ids))

if __name__ == '__main__':
    app.run(debug=True)
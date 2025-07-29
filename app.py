from flask import Flask, render_template, request, jsonify, url_for
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import base64
import os
from PIL import Image
import io
import json
from datetime import datetime
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Configuration Transformers
MODEL_NAME = os.getenv('MODEL_NAME', 'google/medgemma-4b-it')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_LOADED = False
model = None
tokenizer = None
text_generator = None

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_model():
    """Charge le modèle MedGemma avec Transformers"""
    global model, tokenizer, text_generator, MODEL_LOADED
    
    try:
        print(f"🔄 Chargement du modèle {MODEL_NAME} sur {DEVICE}...")
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # Configuration du modèle selon le device
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
        }
        
        if DEVICE == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        # Créer le pipeline de génération de texte
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id
        )
        
        MODEL_LOADED = True
        print(f"✅ Modèle {MODEL_NAME} chargé avec succès sur {DEVICE}")
        
    except Exception as e:
        MODEL_LOADED = False
        print(f"❌ Erreur lors du chargement du modèle: {str(e)}")

# Charger le modèle au démarrage dans un thread séparé
print("🚀 Démarrage de l'application MedExplain...")
model_thread = threading.Thread(target=load_model)
model_thread.start()

def is_medical_emergency(text):
    """Détecte les urgences médicales dans le texte"""
    emergency_keywords = [
        'infarctus', 'crise cardiaque', 'avc', 'accident vasculaire',
        'douleur thoracique', 'essoufflement sévère', 'perte de conscience',
        'convulsions', 'hémorragie', 'fracture ouverte', 'urgence'
    ]
    
    text_lower = text.lower()
    for keyword in emergency_keywords:
        if keyword in text_lower:
            return True
    return False

def generate_medical_response(question, is_image_analysis=False):
    """Génère une réponse médicale avec Transformers"""
    if not MODEL_LOADED or text_generator is None:
        return "⏳ Le modèle est en cours de chargement. Veuillez patienter quelques instants et réessayer."
    
    try:
        # Préparer le prompt médical
        system_prompt = """Tu es un assistant médical IA éducatif basé sur MedGemma. 
        Fournis des informations médicales précises et éducatives uniquement.
        Rappelle toujours de consulter un professionnel de santé qualifié.
        Réponds en français de manière claire et structurée."""
        
        if is_image_analysis:
            system_prompt += "\nTu analyses une image médicale. Décris ce que tu observes de manière éducative."
        
        prompt = f"{system_prompt}\n\nQuestion: {question}\n\nRéponse:"
        
        # Générer la réponse
        response = text_generator(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # Extraire seulement la nouvelle partie générée
        generated_text = response[0]['generated_text']
        answer = generated_text.split("Réponse:")[-1].strip()
        
        # Ajouter un avertissement médical
        medical_disclaimer = "\n\n⚠️ **Avertissement médical**: Cette information est fournie à des fins éducatives uniquement. En cas d'urgence, contactez immédiatement les services d'urgence (15 ou 112). Consultez toujours un professionnel de santé qualifié pour un diagnostic et un traitement appropriés."
        
        return answer + medical_disclaimer
        
    except Exception as e:
        return f"❌ Erreur lors de la génération de la réponse: {str(e)}"

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Interface de chat médical"""
    return render_template('chat.html')

@app.route('/image')
def image_analysis():
    """Interface d'analyse d'images"""
    return render_template('image.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint pour les questions textuelles"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question manquante'}), 400
        
        # Générer la réponse avec Transformers
        answer = generate_medical_response(question)
        
        # Détection d'urgence
        is_emergency = is_medical_emergency(question + " " + answer)
        
        return jsonify({
            'answer': answer,
            'is_emergency': is_emergency,
            'model': MODEL_NAME,
            'device': DEVICE,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Endpoint pour l'analyse d'images médicales"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        question = request.form.get('question', 'Analysez cette image médicale et décrivez ce que vous observez.')
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Sauvegarder l'image
        filename = f"medical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Pour cette version, on analyse le texte de la question
        # Note: MedGemma ne supporte pas nativement les images, on se base sur la description
        image_question = f"Analyse d'image médicale demandée: {question}. Fournissez des conseils généraux d'interprétation d'images médicales."
        
        # Générer la réponse avec Transformers
        answer = generate_medical_response(image_question, is_image_analysis=True)
        
        # Détection d'urgence
        is_emergency = is_medical_emergency(question + " " + answer)
        
        return jsonify({
            'answer': answer,
            'image_url': url_for('static', filename=f'uploads/{filename}'),
            'is_emergency': is_emergency,
            'model': MODEL_NAME,
            'device': DEVICE,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Vérification de l'état de l'application"""
    return jsonify({
        'flask': True,
        'transformers': MODEL_LOADED,
        'model': MODEL_NAME,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy' if MODEL_LOADED else 'loading'
    })

if __name__ == '__main__':
    print("🌐 Application Flask démarrée sur http://localhost:5000")
    print("📚 Interface chat: http://localhost:5000/chat")
    print("🖼️ Analyse d'images: http://localhost:5000/image")
    print("❤️ État de santé: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)

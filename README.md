# MedExplain - IA Médicale avec Transformers et MedGemma 4B IT

Une application d'intelligence artificielle médicale utilisant Transformers et MedGemma 4B IT pour l'analyse d'images médicales et réponses aux questions de santé.

## 🎯 Fonctionnalités

- **Interface Web Flask** : Interface moderne avec Tailwind CSS
- **Questions médicales sans image** : Posez directement des questions médicales
- **Analyse d'images médicales** : Upload et analyse avec MedGemma 4B IT via Transformers
- **Réponses contextuelles** : Analyse intelligente des urgences médicales
- **API REST** : Backend Transformers optimisé

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Cloner le repository
git clone https://github.com/Berachem/BeraMind-MedFocus.git
cd BeraMind-MedFocus

# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Se connecter à Hugging Face (requis pour MedGemma)
hf auth login
```

### 2. Démarrage de l'application

```bash
# Démarrer l'application Flask (inclut le backend IA Transformers)
python app.py
```

**Note**: Au premier démarrage, le modèle MedGemma sera téléchargé automatiquement depuis Hugging Face. Cela peut prendre quelques minutes selon votre connexion internet.

### 3. Accès à l'application

- **Interface Web** : http://localhost:5000
- **Questions médicales** : http://localhost:5000/chat
- **Analyse d'images** : http://localhost:5000/image
- **État de santé** : http://localhost:5000/health

## 🧠 Architecture Transformers

L'application utilise maintenant **Transformers** pour une compatibilité optimale :

- **Modèle** : `google/medgemma-4b-it`
- **Bibliothèque** : Hugging Face Transformers
- **Pipeline** : Text-generation optimisé GPU/CPU
- **Performance** : Inférence directe avec mise en cache

## 🎨 Interface Web

Interface moderne Flask avec Tailwind CSS :

- **Design responsive** : Mobile et desktop
- **Upload drag & drop** : Interface intuitive pour les images
- **Chat interface** : Questions-réponses en temps réel
- **Alertes urgences** : Détection automatique des cas critiques

## 📋 Endpoints

### Interface Web

```
GET /              # Page d'accueil
GET /chat          # Interface chat
GET /image         # Interface analyse d'image
GET /health        # État de l'application
```

### API Backend

```
POST /ask               # Questions médicales
POST /analyze_image     # Analyse d'images
GET /health             # État du modèle
```

## 💡 Utilisation

### Questions médicales

1. Accédez à http://localhost:5000/chat
2. Tapez votre question médicale
3. Recevez une réponse contextuelle

### Analyse d'images

1. Accédez à http://localhost:5000/image
2. Glissez-déposez une image médicale
3. Ajoutez une question optionnelle
4. Analysez les résultats

## 🛠️ Configuration

### Variables d'environnement

```bash
HF_TOKEN=your_token_here           # Token Hugging Face (optionnel)
FLASK_PORT=5000                    # Port Flask
FLASK_DEBUG=True                   # Mode debug
MODEL_NAME=google/medgemma-4b-it   # Modèle à utiliser
```

### Optimisations Transformers

L'application détecte automatiquement votre configuration :

- **GPU CUDA** : Utilisation automatique si disponible
- **CPU** : Fallback automatique sur CPU
- **Mémoire** : Optimisation automatique selon le matériel

```python
# Configuration automatique dans app.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {
    "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
    "device_map": "auto" if DEVICE == "cuda" else None
}
```

## 🚨 Détection d'urgences

Le système détecte automatiquement les urgences médicales :

- **Infarctus** : Alerte rouge + numéro d'urgence
- **AVC** : Protocole FAST + recommandations
- **Douleurs thoraciques** : Évaluation prioritaire
- **Autres urgences** : Redirection appropriée

## 📁 Structure du Projet

```
BeraMind-MedFocus/
├── app.py                  # Application Flask principale
├── requirements.txt        # Dépendances Transformers
├── templates/             # Templates HTML
│   ├── base.html          # Template de base
│   ├── index.html         # Page d'accueil
│   ├── chat.html          # Interface chat
│   └── image.html         # Interface images
├── static/               # Fichiers statiques
│   ├── css/              # Styles Tailwind
│   ├── js/               # JavaScript
│   └── uploads/          # Images uploadées
└── README.md
```

## ⚠️ Avertissements Médicaux

- **Usage éducatif uniquement** - Ne remplace pas un médecin
- **Urgences** : Contactez immédiatement les services d'urgence (15/112)
- **Diagnostic** : Consultez un professionnel de santé qualifié
- **Responsabilité** : L'IA ne peut pas établir de diagnostic médical

## 🔍 Tests

```bash
# Vérifier l'application
curl http://localhost:5000/health

# Test questions médicales
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les symptômes de la grippe?"}'

# Test PowerShell (Windows)
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get
```

## 📦 Dépendances

```txt
# Backend IA
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0

# Frontend Web
flask>=3.0.0
requests>=2.31.0

# Utilitaires
pillow>=10.0.0
python-multipart>=0.0.6
huggingface-hub>=0.18.0
```

## 🚀 Déploiement

### Production

```bash
# Installer Gunicorn
pip install gunicorn

# Démarrer avec Gunicorn (recommandé pour production)
gunicorn -w 2 -b 0.0.0.0:5000 app:app --timeout 120

# Ou directement avec Flask (développement)
python app.py
```

### Configuration GPU

```bash
# Vérifier CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Variables d'environnement CUDA (optionnel)
set CUDA_VISIBLE_DEVICES=0  # Windows
export CUDA_VISIBLE_DEVICES=0  # Linux
```

## 🤝 Contribution

1. Fork le projet
2. Créez une branche pour vos modifications
3. Testez avec Transformers et Flask
4. Respectez les avertissements médicaux
5. Créez une Pull Request

## 📄 Licence

MIT License - Voir `LICENSE` pour plus de détails.

---

**MedExplain 3.0** - Powered by Transformers + MedGemma 4B IT 🏥🤖✨

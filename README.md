# YOLO Vision Web (Streamlit + YOLO)

Application web Streamlit pour la Détection, Segmentation, Classification, Pose, OBB (boîtes orientées) et Tracking avec Ultralytics YOLO (v8/11).
Fonctionne avec images, vidéos, photo depuis la caméra du navigateur et webcam en direct (WebRTC).

✨ Fonctionnalités

Tâches : detect, segment, classify, pose, obb, track

Sources :

Téléversement d’images et de vidéos

Caméra (photo) via st.camera_input (aucune dépendance supplémentaire)

Webcam en direct avec streamlit-webrtc (nécessite av)

Modèles par défaut : famille YOLO11 (repli sur YOLOv8 si absent)

Sorties :

Aperçu annoté dans l’interface

Fichiers générés dans runs/<task>/predict*

Performance :

CPU par défaut (modèles “n” rapides)

Option GPU (CUDA) en sélectionnant device = "0" dans la barre latérale

📁 Arborescence du projet
.
├─ app.py                 # Interface web Streamlit
├─ vision_app.py          # Script CLI unifié (optionnel)
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ uploads/               # Fichiers téléversés (à ne pas versionner)
└─ runs/                  # Résultats annotés (à ne pas versionner)


Ne versionnez pas les poids .pt, runs/, uploads/ ni le dossier .venv/.

🧰 Prérequis

Python 3.9+ (recommandé : 3.10 ou 3.11)

Windows / macOS / Linux

(Optionnel) CUDA pour l’utilisation GPU

Dépendances (requirements.txt)
streamlit
ultralytics>=8.3.0
opencv-python
pillow
lapx
imageio
imageio-ffmpeg
streamlit-webrtc
av


Si vous n’utilisez pas la webcam en direct, vous pouvez omettre streamlit-webrtc et av.

🚀 Installation & exécution locale
Windows (PowerShell ou Git Bash)
# 1) Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/<votre-depot>.git
cd "<votre-depot>"

# 2) Créer et activer un environnement virtuel
python -m venv .venv
# PowerShell :
. .venv\Scripts\Activate.ps1
# Git Bash :
# source .venv/Scripts/activate

# 3) Installer les dépendances
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Lancer l’application
python -m streamlit run app.py


L’application s’ouvre dans votre navigateur (par défaut http://localhost:8501).

🖱️ Utilisation

Lancez python -m streamlit run app.py.

Dans la barre latérale :

Choisissez la Tâche : detect, segment, classify, pose, obb, track

Réglez le seuil de confiance, le périphérique (CPU ou 0 pour GPU), et le tracker (pour track)

Sélectionnez la Source :

Téléverser un fichier (images/vidéos)

Caméra (photo) (prendre une photo via le navigateur)

Webcam en direct (traitement image par image)

Cliquez sur “Processer” pour les fichiers/photos, ou activez Webcam en direct.

Sorties

Images/Vidéos annotées affichées dans la page

Fichiers sauvegardés dans runs/<task>/predict*

⚙️ Script en ligne de commande (optionnel)

Vous pouvez aussi utiliser le script CLI :

# Détection sur une image
python vision_app.py --task detect --source https://ultralytics.com/images/bus.jpg

# Segmentation
python vision_app.py --task segment --source ./mon_image.jpg

# Classification (dossier)
python vision_app.py --task classify --source ./dossier_images/

# Pose sur une vidéo
python vision_app.py --task pose --source ./video.mp4

# OBB
python vision_app.py --task obb --source ./image_orientee.jpg

# Tracking (vidéo ou webcam locale)
python vision_app.py --task track --source 0 --tracker bytetrack.yaml

🧪 Conseils de performance

Utilisez les modèles “n” (par défaut) sur CPU ; passez à “m/l/x” si vous avez besoin de plus de précision.

En GPU : sélectionnez device = "0" (PyTorch + CUDA requis).

Diminuez le seuil de confiance si vous observez peu de détections.

🧯 Dépannage

bash: streamlit: command not found
Lancez avec python -m streamlit run app.py ou réinstallez streamlit dans votre venv.

ModuleNotFoundError: No module named 'ultralytics'
python -m pip install ultralytics

ImportError: No module named 'av' (Webcam en direct)
python -m pip install --only-binary=:all: av streamlit-webrtc

Webcam non détectée
Autorisez l’accès à la caméra dans le navigateur ; en ligne, il faut du HTTPS. Sur CPU, préférez les modèles “n”.

Vidéo annotée non visible
Vérifiez runs/<task>/predict*/ et l’existence du fichier généré. Rechargez la page.

Port occupé
python -m streamlit run app.py --server.port 8502

☁️ Déploiement
Streamlit Community Cloud

Poussez le dépôt sur GitHub.

Allez sur https://streamlit.io
 → Deploy an app.

Sélectionnez votre dépôt et app.py.

(Optionnel) Ajoutez des variables/“secrets” si nécessaire.

Hugging Face Spaces (Streamlit)

Créez un Space (type Streamlit).

Téléversez app.py et requirements.txt.

La construction et la mise en ligne se font automatiquement.

La webcam en direct via WebRTC nécessite généralement HTTPS (ok sur ces plateformes).
En CPU, préférez des vidéos courtes ; les images sont instantanées.

🔒 Confidentialité

Les fichiers téléversés sont stockés localement dans uploads/ et les sorties dans runs/.

En production publique, pensez à nettoyer périodiquement et à contrôler l’accès.

🧹 .gitignore recommandé
__pycache__/
*.pyc
.venv/
venv/
env/
uploads/
runs/
*.log
*.pt
*.onnx
*.engine
*.tflite
.DS_Store
Thumbs.db

🙌 Crédits

Ultralytics YOLO

Streamlit

streamlit-webrtc

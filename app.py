import os
import re
import base64
from io import BytesIO

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from flask_bcrypt import Bcrypt
from pymongo import MongoClient

from PIL import Image, ImageFile
from pymongo import MongoClient
from dotenv import load_dotenv

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from datetime import datetime
import gdown

# =========================
# UTILS
# =========================

from utils.translate import translate_text
from utils.weather_api import get_weather
from utils.chatbot_ai import ask_bot
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from flask_mail import Mail, Message
import random
from authlib.integrations.flask_client import OAuth
from CNN import CNN

# =========================
# FIX IMAGE ERRORS
# =========================

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()

# =========================
# FLASK APP
# =========================

app = Flask(__name__)


app.secret_key = os.getenv("SECRET_KEY")

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)



# =========================
# OAUTH Google
# =========================

oauth = OAuth(app)

google = oauth.register(

    name='google',

    client_id=os.getenv("GOOGLE_CLIENT_ID"),

    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),

    server_metadata_url=
    'https://accounts.google.com/.well-known/openid-configuration',

    client_kwargs={
        'scope': 'openid email profile'
    }

)

# =========================
# LOGIN + BCRYPT
# =========================

bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# =========================
# MONGODB ATLAS
# =========================


MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["plantai"]

users_collection = db["users"]

history_collection = db["history"]

otp_collection = db["otp_codes"]

print("MongoDB Atlas Connected Successfully")
# =========================
# USER CLASS
# =========================

class User(UserMixin):

    def __init__(self, user_data):

        self.id = str(user_data["_id"])
        self.email = user_data["email"]

# =========================
# LOAD USER
# =========================

@login_manager.user_loader
def load_user(user_id):

    from bson.objectid import ObjectId

    user_data = users_collection.find_one(
        {"_id": ObjectId(user_id)}
    )

    if user_data:
        return User(user_data)

    return None

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# =========================
# LOAD CSV FILES
# =========================

disease_info = pd.read_csv(
    "disease_info.csv",
    encoding="cp1252"
)

supplement_info = pd.read_csv(
    "supplement_info.csv",
    encoding="cp1252"
)

# =========================
# LOAD CLASS LABELS
# =========================

with open("class_labels.txt") as f:

    class_names = f.read().splitlines()

num_classes = len(class_names)

print("Total Classes:", num_classes)


# =========================
# LOAD MODEL
# =========================

model = models.resnet50(weights=None)

model.fc = nn.Linear(
model.fc.in_features,
num_classes
)

MODEL_PATH = "new_model.pt"

MODEL_URL = "https://huggingface.co/eletiaryanreddy/plant-disease-model/resolve/main/new_model.pt"




if not os.path.exists(MODEL_PATH):
    print("Downloading AI Model...")

gdown.download(
    MODEL_URL,
    MODEL_PATH,
    quiet=False
)


# Load Model

device = torch.device("cpu")

model = CNN()
model.load_state_dict(
    torch.load("models/new_model.pt", map_location=device)
)
model.eval()

print("Model Loaded Successfully")


# =========================
# CLEAN NAME FUNCTION
# =========================

def clean_name(name):

    if pd.isna(name):
        return ""

    name = str(name)

    name = name.lower()

    name = re.sub(
        r'[^a-z0-9 ]',
        ' ',
        name
    )

    name = " ".join(name.split())

    return name

# =========================
# PREDICTION FUNCTION
# =========================

def prediction(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    input_data = TF.to_tensor(image)

    input_data = input_data.unsqueeze(0)

    with torch.no_grad():

        output = model(input_data)

        probabilities = torch.nn.functional.softmax(
            output[0],
            dim=0
        )

        confidence, predicted = torch.max(
            probabilities,
            0
        )

    disease_name = class_names[predicted.item()]

    confidence_score = round(
        confidence.item() * 100,
        2
    )

    return disease_name, confidence_score

class User(UserMixin):

    def __init__(self, user_data):

        self.id = str(user_data["_id"])

        self.email = user_data["email"]

        self.role = user_data.get(
            "role",
            "farmer"
        )

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home_page():

    return render_template("home.html")

# =========================
# CONTACT PAGE
# =========================

@app.route("/contact")
def contact():

    return render_template("contact-us.html")

# =========================
# AI ENGINE
# =========================

@app.route("/index")
@login_required
def ai_engine_page():

    return render_template("index.html")

# =========================
# MOBILE PAGE
# =========================

@app.route("/mobile-device")
def mobile_device_detected_page():

    return render_template("mobile-device.html")

# =========================
# WEATHER PAGE
# =========================

@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather_page():

    weather_data = None

    if request.method == "POST":

        city = request.form.get("city")

        weather_data = get_weather(city)

    return render_template(
        "weather.html",
        weather=weather_data
    )

# =========================
# VOICE PAGE
# =========================

@app.route("/voice")
@login_required
def voice_page():

    return render_template("voice.html")

# =========================
# CAMERA PAGE
# =========================

@app.route("/camera")
@login_required
def camera_page():

    return render_template("camera.html")

# =========================
# DASHBOARD PAGE
# =========================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user=current_user
    )

# =========================
# HISTORY PAGE
# =========================

@app.route("/history")
@login_required
def history():

    records = history_collection.find({

        "user_email": current_user.email

    }).sort("date", -1)

    return render_template(

        "history.html",

        records=records

    )
# =========================
# CHATBOT
# =========================

      
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():

    if request.method == "POST":

        question = request.form["question"]

        answer = ask_bot(question)

        return answer

    return render_template("chatbot.html")

# =========================
# SUBMIT ROUTE
# =========================

@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit():

    if request.method == 'POST':

        image = request.files['image']

        filename = image.filename

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        image.save(file_path)

        # =========================
        # AI PREDICTION
        # =========================

        pred, confidence = prediction(file_path)

        print("Prediction:", pred)
        print("Confidence:", confidence)

        # =========================
        # DEFAULT VALUES
        # =========================

        description = "No description available"

        prevent = "No prevention available"

        image_url = file_path

        supplement_name = ""

        supplement_image = ""

        buy_link = ""

        # =========================
        # CLEAN PREDICTION
        # =========================

        pred_clean = clean_name(pred)

        print("Clean Prediction:", pred_clean)

        # =========================
        # CLEAN CSV COLUMNS
        # =========================

        disease_info['match_name'] = disease_info[
            'disease_name'
        ].apply(clean_name)

        supplement_info['match_name'] = supplement_info[
            'disease_name'
        ].apply(clean_name)

        # =========================
        # DISEASE MATCH
        # =========================

        disease_match = disease_info[
            disease_info['match_name']
            .str.contains(pred_clean, na=False)
        ]

        if not disease_match.empty:

            row = disease_match.iloc[0]

            description = row.get(
                'description',
                "No description available"
            )

            prevent = row.get(
                'Possible Steps',
                "No prevention available"
            )

            if 'image_url' in row:
                image_url = row['image_url']

        # =========================
        # SUPPLEMENT MATCH
        # =========================

        supplement_match = supplement_info[
            supplement_info['match_name']
            .str.contains(pred_clean, case=False, na=False)
        ]
        # If no match found
        if supplement_match.empty:
            for word in pred_clean.split():
                supplement_match = supplement_info[
                    supplement_info['match_name']
                    .str.contains(word, case=False, na=False)
                ]
                if not supplement_match.empty:
                    break
        # =========================
        # IF EXACT MATCH FAILS
        # =========================

        if supplement_match.empty:

            supplement_match = supplement_info[
                supplement_info['match_name']
                .apply(lambda x: pred_clean in x)
            ]

        # =========================
        # GET SUPPLEMENT DATA
        # =========================

        if not supplement_match.empty:

            srow = supplement_match.iloc[0]

            supplement_name = str(
                srow.get('supplement name', '')
            )

            supplement_image = str(
                srow.get('supplement image', '')
            )

            buy_link = str(
                srow.get('buy link', '')
            )

            print("Supplement Found")
            print("Supplement:", supplement_name)
            print("Buy Link:", buy_link)

        else:

            print("No Supplement Found")

        # =========================
        # TRANSLATION
        # =========================

        language = request.form.get(
            "language",
            "en"
        )

        try:

            if language != "en":

                description = translate_text(
                    description,
                    language
                )

                prevent = translate_text(
                    prevent,
                    language
                )

        except Exception as e:

            print("Translation Error:", e)

        # =========================
        # SHOW SUPPLEMENT
        # =========================

        show_supplement = confidence >= 40
        
        # =========================
        # SAVE HISTORY
        # =========================

        history_collection.insert_one({
            "user_email": current_user.email,
            "disease": pred,
            "confidence": confidence,
            "description": description,
            "prevention": prevent,
            "image": file_path,
            "supplement": supplement_name,
            "buy_link": buy_link,
            "date": datetime.now()
        })

        # =========================
        # RETURN PAGE
        # =========================

        return render_template(

            'submit.html',

            title=pred,

            desc=description,

            prevent=prevent,

            confidence=confidence,

            image_url=image_url,

            uploaded_image=file_path,

            sname=supplement_name,

            simage=supplement_image,

            buy_link=buy_link,

            show_supplement=show_supplement
        )

    return render_template("index.html")


# =========================
# CAMERA PREDICT
# =========================

@app.route("/camera_predict", methods=["POST"])
@login_required
def camera_predict():

    image_data = request.form['image_data']

    image_data = image_data.split(",")[1]

    image = Image.open(
        BytesIO(
            base64.b64decode(image_data)
        )
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        "camera_image.png"
    )

    image.save(file_path)

    pred, confidence = prediction(file_path)

    return render_template(
        "submit.html",
        title=pred,
        confidence=confidence,
        uploaded_image=file_path
    )

# =========================
# MARKET PAGE
# =========================

@app.route('/market')
@login_required
def market():

    supplement_info = pd.read_csv("supplement_info.csv")

    supplement_name = supplement_info['supplement name']
    supplement_image = supplement_info['supplement image']
    disease = supplement_info['disease_name']
    buy = supplement_info['buy link']
    
    return render_template(

        "market.html",

        supplement_image=list(
            supplement_info["supplement image"]
        ),

        supplement_name=list(
            supplement_info["supplement name"]
        ),

        disease=list(
            supplement_info["disease_name"]
        ),

        buy=list(
            supplement_info["buy link"]
        )
    )


# =========================
# SIGNUP PAGE
# =========================


@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]

        # =========================
        # EMAIL VALIDATION
        # =========================

        try:

            valid = validate_email(email)

            email = valid.email

        except EmailNotValidError:

            flash("Invalid Email")

            return redirect("/signup")

        # =========================
        # PHONE VALIDATION
        # =========================

        try:

            phone_obj = phonenumbers.parse(phone, "IN")

            if not phonenumbers.is_valid_number(phone_obj):

                flash("Invalid Phone Number")

                return redirect("/signup")

        except:

            flash("Invalid Phone Number")

            return redirect("/signup")

        # =========================
        # PASSWORD VALIDATION
        # =========================

        if len(password) < 6:

            flash("Password must be at least 6 characters")

            return redirect("/signup")

        # =========================
        # CHECK EXISTING USER
        # =========================

        existing_user = users_collection.find_one({

            "$or": [

                {"email": email},
                {"phone": phone}

            ]

        })

        if existing_user:

            flash("User already exists")

            return redirect("/signup")

        # =========================
        # HASH PASSWORD
        # =========================

        hashed_password = bcrypt.generate_password_hash(
            password
        ).decode("utf-8")

        # =========================
        # GENERATE OTP
        # =========================

        otp = str(random.randint(100000, 999999))

        # =========================
        # SAVE USER
        # =========================

        users_collection.insert_one({

            "email": email,
            "phone": phone,
            "password": hashed_password,
            "role": "farmer",
            "verified": False

        })

        # =========================
        # SAVE OTP
        # =========================

        otp_collection.insert_one({

            "email": email,
            "otp": otp

        })

        # =========================
        # SEND EMAIL
        # =========================

        try:

            msg = Message(

                "PlantAI OTP Verification",

                sender=os.getenv("MAIL_USERNAME"),

                recipients=[email]

            )

            msg.body = f"Your PlantAI OTP is: {otp}"

            mail.send(msg)

            flash("OTP Sent Successfully")

        except Exception as e:

            print(e)

            flash("OTP Sending Failed")

            return redirect("/signup")

        return redirect(
            url_for(
                "verify_otp",
                email=email
            )
        )

    return render_template("signup.html")


# =========================
# VERIFY OTP
# =========================

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():

    email = request.args.get("email")

    if request.method == "POST":

        email = request.form.get("email")
        otp = request.form.get("otp")

        record = otp_collection.find_one({

            "email": email,
            "otp": otp

        })

        if record:

            users_collection.update_one(

                {"email": email},

                {

                    "$set": {

                        "verified": True

                    }

                }

            )

            otp_collection.delete_one({

                "_id": record["_id"]

            })

            flash("Account Verified Successfully")

            return redirect("/login")

        else:

            flash("Invalid OTP")

    return render_template(
        "verify-otp.html",
        email=email
    )



# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email_or_phone = request.form["email_or_phone"]

        password = request.form["password"]

        user_data = users_collection.find_one({

            "$or": [

                {"email": email_or_phone},
                {"phone": email_or_phone}

            ]

        })

        if not user_data:

            flash("User Not Found")

            return redirect("/login")

        # =========================
        # EMAIL VERIFIED CHECK
        # =========================

        if not user_data.get("verified"):

            flash("Please verify your email first")

            return redirect(
                url_for(
                    "verify_otp",
                    email=user_data["email"]
                )
            )

        # =========================
        # PASSWORD CHECK
        # =========================

        try:

            if bcrypt.check_password_hash(

                user_data["password"],
                password

            ):

                user = User(user_data)

                login_user(user)

                flash("Login Successful")

                return redirect("/dashboard")

            else:

                flash("Invalid Password")

        except:

            flash("Password Error")

        return redirect("/login")

    return render_template("login.html")

                    
# =========================
# Google Login
# =========================

@app.route('/google-login')
def google_login():

    redirect_uri = url_for(

        'google_authorized',

        _external=True

    )

    return google.authorize_redirect(
        redirect_uri
    )



# =========================
# Google Authorization
# =========================

@app.route('/google-authorized')
def google_authorized():

    token = google.authorize_access_token()

    user_info = token['userinfo']

    email = user_info['email']

    existing_user = users_collection.find_one({

        "email": email

    })

    # CREATE USER IF NOT EXISTS

    if not existing_user:

        users_collection.insert_one({

            "email": email,

            "password": "",

            "role": "farmer",

            "verified": True

        })

        existing_user = users_collection.find_one({

            "email": email

        })

    user = User(existing_user)

    login_user(user)

    return redirect("/dashboard")

# =========================
# ADMIN
# =========================


@app.route("/admin")
@login_required
def admin():

    if current_user.role != "admin":

        return "Access Denied"

    users = users_collection.find()

    history = history_collection.find()

    total_users = users_collection.count_documents({})

    total_predictions = history_collection.count_documents({})

    return render_template(

        "admin.html",

        users=users,

        history=history,

        total_users=total_users,

        total_predictions=total_predictions

    )

# =========================
# LOGOUT
# =========================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect("/login")

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True,
    threaded=True
)
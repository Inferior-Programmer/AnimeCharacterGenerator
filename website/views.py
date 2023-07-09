from flask import Blueprint, render_template



views = Blueprint('views', __name__)




@views.route('/')
def home():
    return render_template("land.html")

@views.route('/about')
def about():
    return render_template("about.html")

@views.route('/contact')
def contact():
    return render_template("contact.html")

@views.route('/main')
def main():
    return render_template("main.html")

@views.route('/profile')
def profile():
    return render_template("profile.html")




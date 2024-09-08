from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests


#Setting up a Flask app, and configuring a database connection for the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///games.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

#Creating a Game object
class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float)
    description = db.Column(db.Text)
    thumbnail = db.Column(db.String(200))
    featured = db.Column(db.Boolean, default=False)

    #What the object will look like when printed
    def __repr__(self):
        return f'<Game {self.name}>'

#Creating a User object for logins and logouts
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
#Creating an active application context
with app.app_context():
    db.create_all()


#------------------------------------ USER LOGIN/LOGOUT -----------------------------#
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = 'remember' in request.form
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('signup'))

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

#------------------------------------- PURCHASES -------------------------------------#

@app.route('/purchases')
@login_required
def purchases():
    return render_template('purchases.html')


#------------------------------------- HOMEPAGE -------------------------------------#

@app.route('/')
def home():
    return render_template('index.html')


#----------------------------------- ADMIN PANEL ----------------------------------#
@app.route('/admin')
def admin_panel():
    games = Game.query.all()
    return render_template('admin.html', games=games)

@app.route('/get_game_details')
def get_game_details():
    app_id = request.args.get('app_id')
    game = Game.query.filter_by(app_id=app_id).first()
    if game:
        return jsonify({
            'success': False,
            'message': 'This game is already in the database.'
        })

    api_url = f'https://store.steampowered.com/api/appdetails?appids={app_id}'
    response = requests.get(api_url)
    data = response.json()[app_id]['data']

    new_game = Game(
        app_id=app_id,
        name=data['name'],
        price=data['price_overview']['final'] / 100 if 'price_overview' in data else 0,
        description=data['short_description'],
        thumbnail=data['header_image']
    )
    db.session.add(new_game)
    db.session.commit()

    return jsonify({
        'success': True,
        'app_id': app_id,
        'name': data['name'],
        'price': data['price_overview']['final'] / 100 if 'price_overview' in data else 0,
        'description': data['short_description'],
        'thumbnail': data['header_image'],
        'featured': False
    })

@app.route('/toggle_featured', methods=['POST'])
def toggle_featured():
    app_id = request.json['app_id']
    game = Game.query.filter_by(app_id=app_id).first()
    if game:
        if not game.featured:
            # Check if we're trying to feature a new game
            featured_count = Game.query.filter_by(featured=True).count()
            if featured_count >= 12:
                return jsonify({'success': False, 'message': 'Cannot feature more than 12 games at a time.'})
            
        game.featured = not game.featured
        db.session.commit()
        return jsonify({'success': True, 'featured': game.featured})
    return jsonify({'success': False, 'message': 'Game not found'})
    
@app.route('/remove_game', methods=['POST'])
def remove_game():
    app_id = request.json['app_id']
    # Here you would typically remove the game from your database
    # For this example, we'll just return a success response
    return jsonify({'success': True})

#---------------------------------------- STORE ------------------------------------------------
@app.route('/store')
def store():
    games = Game.query.all()
    return render_template('store.html', games=games)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    app_id = request.json['app_id']
    game = Game.query.filter_by(app_id=app_id).first()
    if game:
        if 'cart' not in session:
            session['cart'] = {}
        if not isinstance(session['cart'], dict):
            session['cart'] = {}
        if app_id in session['cart']:
            session['cart'][app_id] += 1
        else:
            session['cart'][app_id] = 1
        session.modified = True
        return jsonify({'success': True, 'message': 'Game added to cart'})
    return jsonify({'success': False, 'message': 'Game not found'})

#---------------------------------------- ABOUT ------------------------------------------------
@app.route('/about')
def about():
    return render_template('about.html')

#---------------------------------------- PROFILE ------------------------------------------------
@app.route('/profile')
def profile():
    return render_template('profile.html')

#---------------------------------------- WISHLIST ------------------------------------------------
@app.route('/wishlist')
@login_required
def wishlist():
    return render_template('wishlist.html')

#------------------------------------------ CART ------------------------------------------------
@app.route('/update_cart', methods=['POST'])
def update_cart():
    app_id = request.json['app_id']
    quantity = request.json['quantity']
    if 'cart' in session and app_id in session['cart']:
        if quantity > 0:
            session['cart'][app_id] = quantity
        else:
            session['cart'].pop(app_id, None)
        session.modified = True
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    session['cart'] = {}
    session.modified = True
    return jsonify({'success': True})

@app.route('/cart')
def cart():
    if 'cart' not in session:
        session['cart'] = {}

    # Convert list to dictionary if necessary
    if isinstance(session['cart'], list):
        cart_dict = {}
        for app_id in session['cart']:
            if app_id in cart_dict:
                cart_dict[app_id] += 1
            else:
                cart_dict[app_id] = 1
        session['cart'] = cart_dict
        session.modified = True

    cart_items = []
    total = 0
    for app_id, quantity in session['cart'].items():
        game = Game.query.filter_by(app_id=app_id).first()
        if game:
            item_total = game.price * quantity
            total += item_total
            cart_items.append({
                'game': game,
                'quantity': quantity,
                'item_total': item_total
            })
    return render_template('cart.html', cart_items=cart_items, total=total)

@app.route('/get_cart_count')
def get_cart_count():
    if 'cart' not in session:
        return jsonify({'count': 0})
    
    if isinstance(session['cart'], list):
        # If cart is still a list, count the number of items
        count = len(session['cart'])
    else:
        # If cart is a dictionary, sum up the quantities
        count = sum(session['cart'].values())
    
    return jsonify({'count': count})

# @app.route('/debug/schema')
# def debug_schema():
#     tables = db.engine.table_names()
#     columns = {table: [col.name for col in db.engine.execute(f'SELECT * FROM {table}').keys()] for table in tables}
#     return jsonify(columns)

if __name__ == '__main__':
    app.run(debug=True)
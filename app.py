from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import requests
import re
from difflib import get_close_matches
import random

# from ai_recommendation_engine import get_recommendations_main  #This is the AI model
import ai_recommendation_engine
from scraping import list_of_game_urls #need scraping.py file for this import



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
    wishlisted_by = db.relationship('User', secondary='wishlist', back_populates='wishlist')
    purchases = db.relationship('Purchase', back_populates='game')

    #What the object will look like when printed
    def __repr__(self):
        return f'<Game {self.name}>'

#Creating a User object for logins and logouts
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    wishlist = db.relationship('Game', secondary='wishlist', back_populates='wishlisted_by')
    purchases = db.relationship('Purchase', back_populates='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'), nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'game_id', name='_user_game_uc'),)
    

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'), nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    price = db.Column(db.Float, nullable=False)
    user = db.relationship('User', back_populates='purchases')
    game = db.relationship('Game', back_populates='purchases')

#Creating an active application context
with app.app_context():
    db.create_all()


#------------------------------------ PAYMENTS -----------------------------#

@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    if 'cart' not in session or not session['cart']:
        flash('Your cart is empty.', 'error')
        return redirect(url_for('cart'))
    
    print("Cart contents:", session['cart'])  # Debugging line
    
    total_amount = 0
    valid_cart_items = {}
    
    for item_id, quantity in session['cart'].items():
        print(f"Checking item_id: {item_id}")  # Debugging line
        
        # Try to get the game by app_id first
        game = Game.query.filter_by(app_id=str(item_id)).first()
        
        if game is None:
            # If not found by app_id, try by database ID
            game = Game.query.get(item_id)
        
        if game:
            print(f"Found game: {game.name}")  # Debugging line
            total_amount += game.price * quantity
            valid_cart_items[item_id] = quantity
        else:
            print(f"Game not found for id: {item_id}")  # Debugging line
            flash(f'A game in your cart (ID: {item_id}) is no longer available and has been removed.', 'warning')
    
    # Update the cart to remove any invalid items
    session['cart'] = valid_cart_items
    session.modified = True
    
    print("Updated cart contents:", session['cart'])  # Debugging line
    
    if not valid_cart_items:
        flash('Your cart is now empty as all items were unavailable.', 'error')
        return redirect(url_for('cart'))
    
    if request.method == 'POST':
        # If it's a POST request, render the payment page
        return render_template('payment.html', total_amount=total_amount)
    
    # If it's a GET request, redirect to cart
    return redirect(url_for('cart'))

@app.route('/process_payment', methods=['POST'])
@login_required
def process_payment():
    payment_method = request.form.get('payment_method')
    flash('Payment successful!', 'success')
    return redirect(url_for('complete_purchase'))

@app.route('/complete_purchase')
@login_required
def complete_purchase():
    if 'cart' not in session or not session['cart']:
        flash('Your cart is empty.', 'error')
        return redirect(url_for('cart'))
   
    total_price = 0
    purchased_games = []

    try:
        for app_id, quantity in session['cart'].items():
            game = Game.query.filter_by(app_id=app_id).first()
            if game:
                purchase = Purchase(
                    user_id=current_user.id,
                    game_id=game.id,
                    price=game.price,
                    purchase_date=datetime.utcnow()
                )
                db.session.add(purchase)
                total_price += game.price * quantity
                purchased_games.append(game.name)

        db.session.commit()
        session['cart'] = {}  # Clear the cart
        flash(f'Thank you for your purchase of {", ".join(purchased_games)}! Total: ${total_price:.2f}', 'success')

    except Exception as e:
        db.session.rollback()
        flash('An error occurred during the purchase. Please try again.', 'error')
        app.logger.error(f'Purchase error: {str(e)}')

    return redirect(url_for('purchases'))

#------------------------------------ USER LOGIN/LOGOUT -----------------------------#
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need to be logged in as an admin to view this page.', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = 'remember' in request.form
        
        if username == "admin" and password == "123":
            user = User.query.filter_by(username="admin").first()
            if not user:
                user = User(username="admin", is_admin=True)
                user.password_hash = "123"  # In a real app, use proper password hashing
                db.session.add(user)
                db.session.commit()
            login_user(user, remember=remember)
            flash('Logged in as admin.', 'success')
            return redirect(url_for('admin_panel'))
        else:
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

@app.route('/purchase', methods=['POST'])
@login_required
def purchase():
    if 'cart' not in session or not session['cart']:
        flash('Your cart is empty.', 'error')
        return redirect(url_for('cart'))
    
    for app_id, quantity in session['cart'].items():
        game = Game.query.filter_by(app_id=app_id).first()
        if game:
            # Check if user already owns the game
            if not Purchase.query.filter_by(user_id=current_user.id, game_id=game.id).first():
                purchase = Purchase(user_id=current_user.id, game_id=game.id, price=game.price)
                db.session.add(purchase)
    
    db.session.commit()
    session['cart'] = {}  # Clear the cart
    flash('Thank you for your purchase!', 'success')
    return redirect(url_for('purchases'))

@app.route('/purchases')
@login_required
def purchases():
    user_purchases = Purchase.query.filter_by(user_id=current_user.id).order_by(Purchase.purchase_date.desc()).all()
    return render_template('purchases.html', purchases=user_purchases)


#------------------------------------- HOMEPAGE -------------------------------------#

@app.route('/')
def home():
    featured_games = Game.query.filter_by(featured=True).limit(12).all()
    return render_template('index.html', featured_games=featured_games)


#----------------------------------- ADMIN PANEL ----------------------------------#
@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    games = Game.query.all()
    return render_template('admin.html', games=games)

@app.route('/get_game_details')
@login_required
@admin_required
def get_game_details():
    price_options = [5, 10, 20, 30, 40, 50, 60]
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
        price= random.choice(price_options),  # Randomly choose a price
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
@login_required
@admin_required
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
@login_required
@admin_required
def remove_game():
    app_id = request.json['app_id']
    game = Game.query.filter_by(app_id=app_id).first()
    if game:
        db.session.delete(game)
        try:
            db.session.commit()
            return jsonify({'success': True, 'message': 'Game successfully removed'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error removing game: {str(e)}'})
    return jsonify({'success': False, 'message': 'Game not found'})


@app.route('/admin/add_bulk_games', methods=['POST'])
@login_required
@admin_required
def add_bulk_games():
    added_count = 0
    skipped_count = 0
    price_options = [5, 10, 20, 30, 40, 50, 60]
    
    for url in list_of_game_urls:
        match = re.search(r'/app/(\d+)/', url)
        if match:
            app_id = match.group(1)
            
            # Check if a game with this app_id already exists
            existing_game = Game.query.filter_by(app_id=app_id).first()
            if existing_game:
                skipped_count += 1
                continue  # Skip this game if it already exists
            
            # Get game details from Steam API
            api_url = f'https://store.steampowered.com/api/appdetails?appids={app_id}'
            response = requests.get(api_url)
            app_data = response.json().get(app_id, {})
            
            if not app_data.get('success') or 'data' not in app_data:
                skipped_count += 1
                continue  # Skip this game if API doesn't return valid data
            
            data = app_data['data']
            
            new_game = Game(
                app_id=app_id,
                name=data.get('name', 'Unknown'),
                price=random.choice(price_options),  # Randomly choose a price
                description=data.get('short_description', ''),
                thumbnail=data.get('header_image', '')
            )
            db.session.add(new_game)
            added_count += 1
    
    try:
        db.session.commit()
        flash(f'Successfully added {added_count} new games to the database. Skipped {skipped_count} games.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while adding games: {str(e)}', 'error')
    
    return redirect(url_for('admin_panel'))


#---------------------------------------- STORE ------------------------------------------------
@app.route('/store')
@login_required
def store():
    # Get all games
    all_games = Game.query.all()
    all_game_names = [game.name for game in all_games]
    
    # Get user's purchased and wishlisted games
    purchased_games = Purchase.query.filter_by(user_id=current_user.id).all()
    purchased_game_names = [Game.query.get(purchase.game_id).name for purchase in purchased_games]
    
    wishlisted_games = Wishlist.query.filter_by(user_id=current_user.id).all()
    wishlisted_game_names = [Game.query.get(wishlist.game_id).name for wishlist in wishlisted_games]
    
    # Combine purchased and wishlisted game names
    liked_game_names = list(set(purchased_game_names + wishlisted_game_names))
    
    # Get recommendations
    if liked_game_names:
        recommended_game_names = ai_recommendation_engine.get_recommendations_main(
            liked_game_names,
            all_game_names,
            num_recommendations=15
        )
        
        # Match recommended game names to game objects
        recommended_games = []
        for rec_name in recommended_game_names:
            # Use difflib to find the closest match
            matches = get_close_matches(rec_name, all_game_names, n=1, cutoff=0.8)
            if matches:
                matched_name = matches[0]
                matched_game = next((game for game in all_games if game.name == matched_name), None)
                if matched_game:
                    recommended_games.append(matched_game)
            # Break if we have 5 valid recommendations
            if len(recommended_games) >= 5:
                break
    else:
        recommended_games = []

    # Limit to 5 recommendations
    recommended_games = recommended_games[:5]
    
    return render_template('store.html', all_games=all_games, recommended_games=recommended_games)

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
    return render_template('wishlist.html', wishlist=current_user.wishlist)

@app.route('/add_to_wishlist/<int:game_id>', methods=['POST'])
@login_required
def add_to_wishlist(game_id):
    game = Game.query.get_or_404(game_id)
    if game not in current_user.wishlist:
        current_user.wishlist.append(game)
        db.session.commit()
        flash('Game added to wishlist!', 'success')
    else:
        flash('Game is already in your wishlist.', 'info')
    
    # Get the 'next' parameter from the form data, defaulting to 'store' if not provided
    next_page = request.form.get('next', 'store')
    
    # Validate the next parameter to prevent open redirect vulnerability
    if next_page not in ['store', 'home']:
        next_page = 'store'
    
    return redirect(url_for(next_page))

@app.route('/remove_from_wishlist/<int:game_id>', methods=['POST'])
@login_required
def remove_from_wishlist(game_id):
    game = Game.query.get_or_404(game_id)
    if game in current_user.wishlist:
        current_user.wishlist.remove(game)
        db.session.commit()
        flash('Game removed from wishlist.', 'success')
    return redirect(url_for('wishlist'))

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


def recreate_database():
    """
    Function to drop and recreate the database from scratch.
    For testing purposes
    """
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("Database recreated successfully.")


def get_all_games():
    """
    Function to get a list of all games in the DB right now.
    Used to send a list of games to the AI model, so that the model
    can be saved and used later, as the server load is too high
    for it to run on command.
    """
    with app.app_context():
        all_games = Game.query.with_entities(Game.name).all()
        return [game.name for game in all_games]
    

if __name__ == '__main__':
    # recreate_database()   # Uncomment this line to recreate the database
    app.run(debug=True)
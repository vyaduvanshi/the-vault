import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import app

def load_and_preprocess_data(file_path='gamedata.csv'):
    """
    Load and preprocess the game data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing game data. Default is 'gamedata.csv'.

    Returns:
    pandas.DataFrame: Preprocessed game data with columns: userid, game, behavior, hoursplayed.
    """
    # Read CSV file, selecting only the first 4 columns
    steam_raw = pd.read_csv(file_path, usecols=[0,1,2,3], names=['userid', 'game', 'behavior', 'hoursplayed'])
    
    # Convert userid to string type for consistency
    steam_raw['userid'] = steam_raw.userid.astype(str)
    
    return steam_raw

def create_game_index(games):
    """
    Create an index for the list of games.

    Args:
    games (list): List of game names.

    Returns:
    pandas.DataFrame: DataFrame with game names and their corresponding index.
    """
    # Create a DataFrame from the list of games
    games_df = pd.DataFrame(games, columns=['game'])
    
    # Add an index column
    games_df['index_col'] = games_df.index
    
    return games_df

def prepare_training_data(steam_df, games_df, num_users=1000):
    """
    Prepare training data for the Restricted Boltzmann Machine (RBM).

    Args:
    steam_df (pandas.DataFrame): Preprocessed game data.
    games_df (pandas.DataFrame): DataFrame with game index.
    num_users (int): Number of users to include in the training data. Default is 1000.

    Returns:
    list: List of user vectors, where each vector represents the games played by a user.
    """
    train_list = []
    usergroup = steam_df.groupby('userid')
    
    # Iterate through users
    for _, cur in list(usergroup)[:num_users]:
        # Initialize a vector for the current user
        temp = [0] * len(games_df)
        
        # Fill in the vector with hours played for each game
        for _, game in cur.iterrows():
            temp[game['index_col']] = game['hoursplayed']
        
        train_list.append(temp)
    
    return train_list

def create_rbm_model(visible_units, hidden_units=50):
    """
    Create the Restricted Boltzmann Machine (RBM) model using TensorFlow.

    Args:
    visible_units (int): Number of visible units (games).
    hidden_units (int): Number of hidden units. Default is 50.

    Returns:
    tuple: Contains TensorFlow placeholders and operations for the RBM model.
    """
    # Initialize bias vectors and weight matrix
    vb = tf.placeholder(tf.float32, [visible_units])  # Visible units bias
    hb = tf.placeholder(tf.float32, [hidden_units])   # Hidden units bias
    W = tf.placeholder(tf.float32, [visible_units, hidden_units])  # Weight matrix
    
    # Forward pass
    v0 = tf.placeholder("float", [None, visible_units])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
    
    # Backward pass (reconstruction)
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
    
    # Learning rate
    alpha = 1.0
    
    # Compute gradients
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
    
    # Update rules for bias vectors and weight matrix
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)
    
    # Compute reconstruction error
    err = v0 - v1
    err_sum = tf.reduce_mean(err * err)
    
    return v0, W, vb, hb, update_w, update_vb, update_hb, err_sum

def train_rbm(train_list, visible_units, hidden_units=50, epochs=30, batch_size=150):
    """
    Train the Restricted Boltzmann Machine (RBM) model.

    Args:
    train_list (list): List of user vectors for training.
    visible_units (int): Number of visible units (games).
    hidden_units (int): Number of hidden units. Default is 50.
    epochs (int): Number of training epochs. Default is 30.
    batch_size (int): Size of each training batch. Default is 150.

    Returns:
    tuple: Contains trained model parameters and TensorFlow session.
    """
    # Create the RBM model
    v0, W, vb, hb, update_w, update_vb, update_hb, err_sum = create_rbm_model(visible_units, hidden_units)
    
    # Initialize model parameters
    cur_w = np.zeros([visible_units, hidden_units], np.float32)
    cur_vb = np.zeros([visible_units], np.float32)
    cur_hb = np.zeros([hidden_units], np.float32)
    prv_w = np.zeros([visible_units, hidden_units], np.float32)
    prv_vb = np.zeros([visible_units], np.float32)
    prv_hb = np.zeros([hidden_units], np.float32)
    
    # Start TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training loop
    for _ in range(epochs):
        for start, end in zip(range(0, len(train_list), batch_size), range(batch_size, len(train_list), batch_size)):
            batch = train_list[start:end]
            # Update weights and biases
            cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
    
    return sess, v0, W, vb, hb, prv_w, prv_vb, prv_hb



def train_and_save_model(all_games):

    # Load and preprocess data
    steam_raw = load_and_preprocess_data()

    # Create game index
    games_df = create_game_index(all_games)

    # Merge game index with steam data
    steam_df = steam_raw.merge(games_df, on='game')
    
    # Prepare training data
    train_list = prepare_training_data(steam_df, games_df)

    # Train RBM
    visible_units = len(all_games)
    sess, v0, W, vb, hb, prv_w, prv_vb, prv_hb = train_rbm(train_list, visible_units)

    # Create a dictionary to store the saveable variables
    saveable_vars = {}

    # Create new variables with the same shapes as the placeholders
    saveable_vars['W'] = tf.Variable(tf.random.normal([98, 50]), name='W')
    saveable_vars['vb'] = tf.Variable(tf.zeros([98]), name='vb')  # Adjust shape if different
    saveable_vars['hb'] = tf.Variable(tf.zeros([50]), name='hb')  # Adjust shape if different

    # For the numpy arrays, we can directly convert them to TensorFlow Variables
    saveable_vars['prv_w'] = tf.Variable(prv_w, name='prv_w')
    saveable_vars['prv_vb'] = tf.Variable(prv_vb, name='prv_vb')
    saveable_vars['prv_hb'] = tf.Variable(prv_hb, name='prv_hb')

    # Initialize the new variables
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Now create the saver with the saveable variables
    saver = tf.compat.v1.train.Saver(saveable_vars)

    save_path = saver.save(sess, "rbm_model.ckpt")
    print(f"Model saved in path: {save_path}")



# Define the RBM prediction function
def predict_rbm(v0, W, hb, vb):
    hidden_probs = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    hidden_probs_sample = tf.nn.relu(tf.sign(hidden_probs - tf.random.uniform(tf.shape(hidden_probs))))
    visible_probs = tf.nn.sigmoid(tf.matmul(hidden_probs_sample, tf.transpose(W)) + vb)
    return visible_probs



def get_recommendations_main(liked_games, all_games, num_recommendations=6):
   
    # all_games = app.get_all_games() #gets a list of all_games from the web app
    #Adding a manual list here in case the dimension changes in case of unsuccesful API call from the webapp
    all_games = ['Counter-Strike 2',
                'Dota 2',
                'PUBG: BATTLEGROUNDS',
                'Banana',
                'Black Myth: Wukong',
                'NARAKA: BLADEPOINT',
                'Apex Legends™',
                'Satisfactory',
                'Grand Theft Auto V',
                'Rust',
                "Baldur's Gate 3",
                'Warhammer 40,000: Space Marine 2',
                'Once Human',
                'Call of Duty®',
                'War Thunder',
                "Tom Clancy's Rainbow Six® Siege",
                'Team Fortress 2',
                'Stardew Valley',
                'Football Manager 2024',
                "Sid Meier’s Civilization® VI",
                'ELDEN RING',
                'HELLDIVERS™ 2',
                'Warframe',
                'Crab Game',
                'Dead by Daylight',
                'Overwatch® 2',
                'DayZ',
                'Governor of Poker 3',
                'Hearts of Iron IV',
                '7 Days to Die',
                'VRChat',
                'Monster Hunter: World',
                'The Crew™ 2',
                "Don't Starve Together",
                'The Sims™ 4',
                'Cyberpunk 2077',
                'Destiny 2',
                'Forza Horizon 4',
                'God of War Ragnarök',
                'Project Zomboid',
                'Red Dead Redemption 2',
                'Euro Truck Simulator 2',
                'Path of Exile',
                'Cats',
                'Terraria',
                'EA SPORTS FC™ 24',
                'ARK: Survival Evolved',
                'The First Descendant',
                'Yu-Gi-Oh! Master Duel',
                'ARK: Survival Ascended',
                'NBA 2K25',
                'Left 4 Dead 2',
                'Tapple - Idle Clicker',
                'Total War: WARHAMMER III',
                'RimWorld',
                'Farming Simulator 22',
                'Core Keeper',
                'Palworld',
                'FINAL FANTASY XIV Online',
                'Dark and Darker',
                'Mount & Blade II: Bannerlord',
                'The Elder Scrolls V: Skyrim Special Edition',
                "Garry's Mod",
                'FINAL FANTASY XVI',
                'TCG Card Shop Simulator',
                "No Man's Sky",
                'SCUM',
                'Unturned',
                'MIR4',
                'Valheim',
                'BeamNG.drive',
                'Crusader Kings III',
                "Sid Meier's Civilization® V",
                'Lethal Company',
                'Age of Empires II: Definitive Edition',
                'The Binding of Isaac: Rebirth',
                'Hunt: Showdown 1896',
                'Stellaris',
                'Street Fighter™ 6',
                'Forza Horizon 5',
                'Geometry Dash',
                'Eternal Return',
                'Fallout 4',
                'Battlefield™ V',
                'STALCRAFT: X',
                'Slay the Spire',
                'PAYDAY 2',
                'Europa Universalis IV',
                'Supermarket Together',
                'The Witcher 3: Wild Hunt',
                'Phasmophobia',
                'Cookie Clicker',
                'Bloons TD 6',
                'THE FINALS',
                'Killing Floor 2',
                'Hogwarts Legacy',
                'Risk of Rain 2',
                'Frostpunk']

    # Create game_to_index and index_to_game dictionaries
    game_to_index = {game: index for index, game in enumerate(all_games)}
    index_to_game = {index: game for game, index in game_to_index.items()}

    # Create a new session
    new_sess = tf.compat.v1.Session()

    # Create variables with the same names as in your saved model
    W = tf.Variable(tf.zeros((98, 50)), name="W")
    vb = tf.Variable(tf.zeros((98,)), name="vb")
    hb = tf.Variable(tf.zeros((50,)), name="hb")
    prv_w = tf.Variable(tf.zeros((98, 50)), name="prv_w")
    prv_vb = tf.Variable(tf.zeros((98,)), name="prv_vb")
    prv_hb = tf.Variable(tf.zeros((50,)), name="prv_hb")

    # Initialize the variables
    new_sess.run(tf.compat.v1.global_variables_initializer())

    # Create a checkpoint reader
    try:
        reader = tf.train.load_checkpoint("rbm_model.ckpt")
    except:
        reader = tf.train.load_checkpoint("the-vault/rbm_model.ckpt")

    # Get the variable names and shapes in the checkpoint
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("Variables in checkpoint:", var_to_shape_map)

    # Function to load a variable from the checkpoint
    def create_and_load_variable(name):
        shape = var_to_shape_map[name]
        var = tf.compat.v1.get_variable(name, shape=shape, initializer=tf.zeros_initializer())
        tensor = reader.get_tensor(name)
        new_sess.run(tf.compat.v1.assign(var, tensor))
        return var

    # Try to load each variable
    try:
        # Create and load variables
        W = create_and_load_variable("W")
        vb = create_and_load_variable("vb")
        hb = create_and_load_variable("hb")
        prv_w = create_and_load_variable("prv_w")
        prv_vb = create_and_load_variable("prv_vb")
        prv_hb = create_and_load_variable("prv_hb")
    except Exception as e:
        print(f"Error loading variable: {e}")

    # Print shapes of loaded variables
    print("W shape:", new_sess.run(W).shape)
    print("vb shape:", new_sess.run(vb).shape)
    print("hb shape:", new_sess.run(hb).shape)
    print("prv_w shape:", new_sess.run(prv_w).shape)
    print("prv_vb shape:", new_sess.run(prv_vb).shape)
    print("prv_hb shape:", new_sess.run(prv_hb).shape)

    # Define the RBM prediction function
    def predict_rbm(v0):
        hidden_probs = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        hidden_probs_sample = tf.nn.relu(tf.sign(hidden_probs - tf.random.uniform(tf.shape(hidden_probs))))
        visible_probs = tf.nn.sigmoid(tf.matmul(hidden_probs_sample, tf.transpose(W)) + vb)
        return visible_probs

    # Create the recommendation function
    def recommend_games(user_games, top_n=10):
        v0 = tf.compat.v1.placeholder(tf.float32, shape=(1, 98), name='v0')
        predictions = predict_rbm(v0)
        
        input_vector = np.zeros((1, 98))
        for game in user_games:
            if game in game_to_index:
                input_vector[0, game_to_index[game]] = 1

        recommendations = new_sess.run(predictions, feed_dict={v0: input_vector})

        top_indices = np.argsort(recommendations[0])[::-1][:top_n]
        recommended_games = [index_to_game[i] for i in top_indices if index_to_game[i] not in user_games]

        return recommended_games[:top_n]

    # Test the recommendation function
    user_games = liked_games
    recommendations = recommend_games(user_games)
    return recommendations

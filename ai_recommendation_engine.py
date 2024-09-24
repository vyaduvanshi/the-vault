import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

def get_recommendations(sess, v0, W, vb, hb, prv_w, prv_vb, prv_hb, input_user, games_df):
    """
    Generate game recommendations for a given user.

    Args:
    sess (tf.Session): TensorFlow session.
    v0, W, vb, hb: TensorFlow placeholders for the RBM model.
    prv_w, prv_vb, prv_hb: Trained model parameters.
    input_user (list): Vector representing the input user's game preferences.
    games_df (pandas.DataFrame): DataFrame with game index.

    Returns:
    pandas.DataFrame: Sorted DataFrame of games with recommendation scores.
    """
    # Forward pass through the RBM
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = sess.run(hh0, feed_dict={v0: input_user, W: prv_w, hb: prv_hb})
    rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
    
    # Add recommendation scores to the games DataFrame
    games_df["Recommendation Score"] = rec[0]
    return games_df.sort_values(["Recommendation Score"], ascending=False)

def get_recommendations_main(liked_games, all_games, num_recommendations=6):
    """
    Generate game recommendations based on a user's liked games.

    Args:
    liked_games (list): List of games the user likes.
    all_games (list): List of all available games.
    num_recommendations (int): Number of games to recommend. Default is 6.

    Returns:
    list: List of recommended game names.
    """
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
    
    # Create input user vector
    input_user = np.zeros(visible_units)
    for game in liked_games:
        if game in games_df['game'].values:
            index = games_df[games_df['game'] == game]['index_col'].values[0]
            input_user[index] = 1
    
    # Get recommendations
    recommendations = get_recommendations(sess, v0, W, vb, hb, prv_w, prv_vb, prv_hb, [input_user], games_df)
    
    # Filter out games the user already likes
    unplayed_games = recommendations[~recommendations['game'].isin(liked_games)]
    
    return unplayed_games['game'].head(num_recommendations).tolist()

# Example usage:
# liked_games = ["Game1", "Game2", "Game3"]
# all_games = ["Game1", "Game2", "Game3", ..., "GameN"]  # List of all games in the dataset
# recommended_games = recommend_games(liked_games, all_games, num_recommendations=6)
# print(recommended_games)
# the vault

A digital games marketplace with an AI recommendation engine.

To see deployed live version of the project - www.vyaduvanshi.pythonanywhere.com

<br>

## Highlighted Features

### Recommendation Engine in Action

The GIF below demonstrates how the recommendation engine activates dynamically.
When the wishlist is empty, no recommendations are shown.
Once a game is added, the engine instantly analyzes the wishlist data and generates personalized suggestions.

![AI Recommendations in Action](assets/recommendation_engine.gif)

### Home Page

![Storefront Overview](assets/homepage.png)

### Admin Panel
![Admin Dashboard](assets/admin_panel.png)

<br>

## Tech stack used

| Area | Tools |
| --- | --- |
| Frontend | HTML, CSS, JavaScript |
| Web App Backend | Python, Flask, Jinja2 |
| Database | SQLite, SQLAlchemy |
| Web Scraping | Requests, BeautifulSoup |
| Data Manipulation | NumPy, Pandas |
| Graphing | Plotly |
| Artificial Intelligence | Python, Jupyter Notebook, Keras, TensorFlow |

<br>

## How the recommendation engine works (Restricted Boltzmann Machine)

A Restricted Boltzmann Machine (RBM) is a type of neural network that learns hidden patterns in data.
In a recommendation system, RBMs help predict what items a user might enjoy next by learning relationships between users and items.

- Model structure: An RBM is a two-layer neural network with a visible layer (game choices) and a hidden layer (tries to learn *why* the user liked those movies). Each visible unit connects to every hidden unit, but there are no lateral connections within a layer, which simplifies training.
- Training objective: The RBM learns to reconstruct user-game interaction vectors. For this project, each training sample is a binary or weighted vector indicating whether a user played or liked a particular game. During training, Contrastive Divergence adjusts weights so reconstructed vectors resemble the originals.



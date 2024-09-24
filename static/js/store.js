// const addToCartButtons = document.querySelectorAll('.add-to-cart');

function addToCart(appId) {
    fetch('/add_to_cart', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ app_id: appId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Game added to cart!');
            updateCartCounter();
        } else {
            alert('Failed to add game to cart. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
}
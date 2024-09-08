const addToCartButtons = document.querySelectorAll('.add-to-cart');

addToCartButtons.forEach(button => {
    button.addEventListener('click', function() {
        const appId = this.getAttribute('data-id');
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
                alert(data.message);
                updateCartCounter();  // Update the cart counter in the nav bar
            } else {
                alert('Failed to add game to cart');
            }
        });
    });
});

const quantityBtns = document.querySelectorAll('.quantity-btn');
const removeBtns = document.querySelectorAll('.remove-btn');
const clearCartBtn = document.getElementById('clear-cart');

function updateCart(appId, quantity) {
    fetch('/update_cart', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ app_id: appId, quantity: quantity })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            location.reload();  // Refresh the page to update cart
        }
    });
}

quantityBtns.forEach(btn => {
    btn.addEventListener('click', function() {
        const appId = this.getAttribute('data-id');
        const action = this.getAttribute('data-action');
        const quantitySpan = this.parentElement.querySelector('.quantity');
        let quantity = parseInt(quantitySpan.textContent);

        if (action === 'increase') {
            quantity++;
        } else if (action === 'decrease' && quantity > 1) {
            quantity--;
        }

        updateCart(appId, quantity);
    });
});

removeBtns.forEach(btn => {
    btn.addEventListener('click', function() {
        const appId = this.getAttribute('data-id');
        updateCart(appId, 0);  // Set quantity to 0 to remove
    });
});

clearCartBtn.addEventListener('click', function() {
    fetch('/clear_cart', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            location.reload();  // Refresh the page to update cart
        }
    });
});
function toggleDropdown(event) {
    event.preventDefault();
    document.getElementById("profileDropdown").classList.toggle("show");
}

window.onclick = function(event) {
    if (!event.target.matches('#profile-icon')) {
        var dropdowns = document.getElementsByClassName("dropdown-content");
        for (var i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}

function updateCartCounter() {
    fetch('/get_cart_count')
        .then(response => response.json())
        .then(data => {
            const counter = document.getElementById('cart-counter');
            counter.textContent = data.count;
            counter.style.display = data.count > 0 ? 'inline' : 'none';
        });
}

document.addEventListener('DOMContentLoaded', updateCartCounter);


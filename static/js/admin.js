const gameSearch = document.querySelector('.game-search');
const addGameButton = document.querySelector('.add-game');
const gamesTable = document.querySelector('.games-table tbody');
const errorMessage = document.getElementById('error-message');


// Adding event listener to Add button
addGameButton.addEventListener('click', () => {

    const steamLink = gameSearch.value.trim();
    if (steamLink) {
        // Extract the app ID from the Steam link
        const appId = steamLink.match(/\/app\/(\d+)/)[1];
        
        // Make API call to Flask backend
        fetch(`/get_game_details?app_id=${appId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Add the game to the list
                    const newRow = createGameRow(data, gamesTable.children.length + 1);
                    gamesTable.appendChild(newRow);
                    gameSearch.value = '';
                } else {
                    showError(data.message || 'Failed to fetch game details. Please try again.');                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError('An error occurred. Please try again.');
            });
    }
});

// Adding event listener to Remove button
gamesTable.addEventListener('click', function(e) {
    if (e.target.classList.contains('remove-game')) {
        const appId = e.target.getAttribute('data-id');
        removeGame(appId, e.target.closest('tr'));
    }
});

// Adding event listener to Featured checkbox toggle button
gamesTable.addEventListener('change', function(e) {
    if (e.target.classList.contains('featured-checkbox')) {
        const appId = e.target.getAttribute('data-id');
        toggleFeatured(appId, e.target);
    }
});

// Function that creates the HTML code for a game row to be added in Current Games
function createGameRow(game, index) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>${index}</td>
        <td><img src="${game.thumbnail}" alt="${game.name}" width="100"></td>
        <td>${game.name}</td>
        <td>$${game.price.toFixed(2)}</td>
        <td>${game.app_id}</td>
        <td>
            <input type="checkbox" class="featured-checkbox" data-id="${game.app_id}" ${game.featured ? 'checked' : ''}>
        </td>
        <td>
            <button class="remove-game" data-id="${game.app_id}">Remove</button>
        </td>
    `;
    return tr;
}

function removeGame(appId, row) {
    fetch('/remove_game', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ app_id: appId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            row.remove();
            updateRowNumbers();
            showError(data.message); // Show success message
        } else {
            showError(data.message || 'Failed to remove game. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('An error occurred. Please try again.');
    });
}

// Function to toggle the featured checkbox in each row
function toggleFeatured(appId, checkbox) {
    fetch('/toggle_featured', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ app_id: appId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            checkbox.checked = data.featured;
        } else {
            checkbox.checked = !checkbox.checked; // Revert the checkbox
            showError(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        checkbox.checked = !checkbox.checked; // Revert the checkbox
        showError('An error occurred. Please try again.');
    });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000); // Hide the message after 3 seconds
}


function updateRowNumbers() {
    const rows = gamesTable.querySelectorAll('tr');
    rows.forEach((row, index) => {
        row.cells[0].textContent = index + 1;
    });
}
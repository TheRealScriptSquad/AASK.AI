document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('regForm').addEventListener('submit', function(event) {
        event.preventDefault();
        // Fetch data from form and post it to Flask backend
        fetch('/web', {
            method: 'POST',
            body: new FormData(this),
        })
        .then(response => response.text())
        .then(data => {
            // Display result from Flask backend
            //console.log(data)
            document.getElementById("score").value = Number(data);
        })
        .catch(error => console.error('Error:', error));
    });
});
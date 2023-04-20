    // Store a reference to the result element
    const resultElement = document.getElementById("result");

    // Handle form submission
    document.getElementById("predict-form").addEventListener("submit", function(event) {
        event.preventDefault();
        const form = event.target;
        const data = new FormData(form);

        // Send POST request to server to predict melting temperature
        fetch("/predict", {
            method: "POST",
            body: data
        })
        .then(response => response.json())
        .then(data => {
            // Update the temperature information in the result element
            resultElement.innerHTML = `Predicted melting temperature: ${data.melting_temperature.toFixed(2)}°C`;

            // Add CSS animation to result element
            resultElement.classList.add("animate__animated", "animate__fadeIn");
        });
    });

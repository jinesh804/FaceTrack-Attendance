document.getElementById('startRecognition').addEventListener('click', function() {
    fetch('/recognize', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        alert(data.status === "success" ? 
              `Attendance taken for ${data.name} at ${data.timestamp}` : 
              `Error: ${data.message}`);
    })
    .catch(error => console.error('Error:', error));
});

function openCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            var videoElement = document.getElementById('cameraFeed');
            videoElement.srcObject = stream;
            videoElement.play();
            videoElement.style.display = 'block';
        })
        .catch(function (error) {
            console.log('Error accessing camera:', error);
        });

    var captureButton = document.getElementById('capture_image_button');
    captureButton.style.display = 'block';
}



function captureImage() {
    var video = document.getElementById('cameraFeed');
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    var imageData = canvas.toDataURL('image/png'); // Convert canvas to image data URL

    // Display the captured image on the page
    var capturedImageContainer = document.getElementById('capturedImageContainer');
    var imgElement = document.createElement('img');
    imgElement.src = imageData;

    capturedImageContainer.innerHTML = ''; // Clear the container
    capturedImageContainer.appendChild(imgElement); // Append the new imgElement
}





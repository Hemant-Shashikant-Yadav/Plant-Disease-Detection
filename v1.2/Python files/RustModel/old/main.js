function uploadImage() {
    var input = document.getElementById('imageInput');
    var file = input.files[0];

    if (file) {
        var formData = new FormData();
        formData.append('image', file);

        fetch('/process_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('processedImage').src = 'data:image/png;base64,' + data;
            document.getElementById('processedImage').style.display = 'block';
        })
        .catch(error => console.error('Error:', error));
    }
}

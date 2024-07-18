function detectTumor() {
    var input = document.getElementById('imageUpload');
    var file = input.files[0];

    if (file) {
        var reader = new FileReader();
        reader.readAsDataURL(file);

        reader.onloadend = function () {
            var base64data = reader.result;
            
            // Send the base64 data to the server for tumor detection
            // You can make an AJAX request here to your backend

            // Example: Display the result on the page
            var resultDiv = document.getElementById('result');
            resultDiv.innerText = 'Tumor detected!';
        };
    }
}

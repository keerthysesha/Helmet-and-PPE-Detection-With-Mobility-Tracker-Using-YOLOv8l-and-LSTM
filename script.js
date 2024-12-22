document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('output').innerText = 'Processing...';
    document.getElementById('uploadedImage').style.display = 'none';
    document.getElementById('uploadedVideo').style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        const outputElement = document.getElementById('output');
        
        if (result.success) {
            // Apply success class
            outputElement.classList.add('success');
            outputElement.classList.remove('error');
            
            // Display detected counts
            outputElement.innerHTML = `
                Helmets: <span class="count">${result.helmets}</span><br>
                Gloves: <span class="count">${result.gloves}</span><br>
                Masks: <span class="count">${result.masks}</span><br>
                Goggles: <span class="count">${result.goggles}</span>`;

            // Show the processed file
            if (result.type === 'image') {
                document.getElementById('uploadedImage').src = `data:image/jpeg;base64,${result.image}`;
                document.getElementById('uploadedImage').style.display = 'block';
            } else if (result.type === 'video') {
                document.getElementById('uploadedVideo').src = `data:video/mp4;base64,${result.video}`;
                document.getElementById('uploadedVideo').style.display = 'block';
            }
        } else {
            // Apply error class if no success
            outputElement.classList.add('error');
            outputElement.classList.remove('success');
            
            outputElement.innerText = 'Error processing file.';
        }
    } catch (error) {
        const outputElement = document.getElementById('output');
        outputElement.classList.add('error');
        outputElement.classList.remove('success');
        outputElement.innerText = 'Error: Unable to connect to the server.';
    }
});
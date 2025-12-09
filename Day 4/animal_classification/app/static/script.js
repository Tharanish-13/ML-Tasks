// app/static/script.js
document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file');
  const preview = document.getElementById('preview');
  const previewContainer = document.getElementById('preview-container');
  const predictBtn = document.getElementById('predictBtn');
  const modelSelect = document.getElementById('model');
  const resultDiv = document.getElementById('result');

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.src = ev.target.result;
      preview.hidden = false;
    };
    reader.readAsDataURL(file);
  });

  predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
      alert('Please upload an image first.');
      return;
    }
    const model = modelSelect.value;
    const formData = new FormData();
    formData.append('model', model);
    formData.append('file', file, file.name);

    resultDiv.innerHTML = "Predicting...";
    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        resultDiv.innerHTML = `<b>Error:</b> ${data.error || JSON.stringify(data)}`;
        return;
      }
      const confList = data.confidences ? Object.entries(data.confidences)
        .sort((a,b)=> b[1]-a[1])
        .slice(0,5)
        .map(([k,v]) => `<li>${k}: ${(v*100).toFixed(1)}%</li>`).join('') : '';

      resultDiv.innerHTML = `
        <strong>Prediction:</strong> ${data.prediction} <br/>
        <strong>Class id:</strong> ${data.class_id}
        ${confList ? `<br/><strong>Top confidences:</strong><ul>${confList}</ul>` : ''}
      `;
    } catch (err) {
      resultDiv.innerHTML = `<b>Network error:</b> ${err}`;
    }
  });
});

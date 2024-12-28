// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('prediction-form');
  const imageInput = document.getElementById('image-input');
  const previewImage = document.getElementById('preview-image');
  const loading = document.querySelector('.loading');
  const resultBox = document.getElementById('prediction-result');
  const sentimentResult = document.getElementById('sentiment-result');

  // Preview image when selected
  imageInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
          const reader = new FileReader();
          reader.onload = function(e) {
              previewImage.src = e.target.result;
              previewImage.classList.remove('d-none');
          }
          reader.readAsDataURL(file);
      }
  });

  // Handle form submission
  form.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const formData = new FormData();
      formData.append('text', document.getElementById('text-input').value);
      formData.append('image', imageInput.files[0]);

      // Show loading, hide results
      loading.classList.remove('d-none');
      resultBox.classList.add('d-none');

      try {
          const response = await fetch('/predict', {
              method: 'POST',
              body: formData
          });
          const data = await response.json();

          if (data.success) {
              // Update sentiment result with appropriate styling
              sentimentResult.textContent = data.sentiment;
              sentimentResult.className = `ms-2 badge bg-${getSentimentColor(data.sentiment)}`;
              
              // Update probability bars
              updateProbabilityBars(data.probabilities);
              
              // Show results
              resultBox.classList.remove('d-none');
              
              // Smooth scroll to results
              resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          } else {
              showError('Error: ' + data.error);
          }
      } catch (error) {
          showError('Error: ' + error.message);
      } finally {
          loading.classList.add('d-none');
      }
  });

  function updateProbabilityBars(probabilities) {
      for (const [sentiment, probability] of Object.entries(probabilities)) {
          const progressBar = document.querySelector(`#${sentiment}-probability .progress-bar`);
          const percentage = (probability * 100).toFixed(1);
          progressBar.style.width = `${percentage}%`;
          progressBar.textContent = `${percentage}%`;
          
          // Update aria attributes for accessibility
          progressBar.setAttribute('aria-valuenow', percentage);
          progressBar.setAttribute('aria-valuemin', '0');
          progressBar.setAttribute('aria-valuemax', '100');
      }
  }

  function getSentimentColor(sentiment) {
      const colors = {
          'Negative': 'danger',
          'Neutral': 'warning',
          'Positive': 'success'
      };
      return colors[sentiment] || 'primary';
  }

  function showError(message) {
      // Create error element if it doesn't exist
      let errorDiv = document.querySelector('.error-message');
      if (!errorDiv) {
          errorDiv = document.createElement('div');
          errorDiv.className = 'error-message';
          form.appendChild(errorDiv);
      }
      
      errorDiv.textContent = message;
      errorDiv.style.display = 'block';
      
      // Hide error after 5 seconds
      setTimeout(() => {
          errorDiv.style.display = 'none';
      }, 5000);
  }
});
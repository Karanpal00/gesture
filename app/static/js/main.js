/**
 * Main JavaScript for the Gesture & Face Authentication API frontend
 */

// Check if API is available on page load
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    
    if (data.status === 'ok') {
      console.log('API is operational');
    } else {
      showAlert('API may not be functioning correctly. Check server logs.');
    }
  } catch (error) {
    console.error('Error checking API health:', error);
    showAlert('Unable to connect to API. Please ensure the server is running.');
  }
});

/**
 * Show an alert message
 * @param {string} message - The message to display
 * @param {string} type - The alert type (success, danger, warning, info)
 */
function showAlert(message, type = 'danger') {
  const alertContainer = document.createElement('div');
  alertContainer.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
  alertContainer.style.zIndex = '9999';
  alertContainer.role = 'alert';
  
  alertContainer.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  document.body.appendChild(alertContainer);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    const bsAlert = new bootstrap.Alert(alertContainer);
    bsAlert.close();
  }, 5000);
}

/**
 * Toggle sections visibility
 * @param {string} sectionId - The ID of the section to show
 */
function showSection(sectionId) {
  const sections = document.querySelectorAll('.api-section');
  sections.forEach(section => {
    section.style.display = section.id === sectionId ? 'block' : 'none';
  });
}

/**
 * Format API response for display
 * @param {Object} response - The API response
 * @param {string} status - The HTTP status
 * @returns {string} HTML string to display
 */
function formatResponse(response, status) {
  return `
    <div class="response-container mt-3">
      <div class="d-flex justify-content-between">
        <h6>Response (${status})</h6>
        <span class="badge ${status.startsWith('2') ? 'bg-success' : 'bg-danger'}">${status}</span>
      </div>
      <pre class="bg-dark text-light p-3 rounded"><code>${JSON.stringify(response, null, 2)}</code></pre>
    </div>
  `;
}

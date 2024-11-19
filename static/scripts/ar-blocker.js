function checkAspectRatio() {
  const overlay = document.getElementById('overlay');
  const content = document.getElementById('content');

  // Check if the aspect ratio matches a mobile device (height > width)
  if (window.innerHeight > window.innerWidth) {
    overlay.style.display = 'none'; // Hide the overlay
    content.style.display = 'block'; // Show the content
  } else {
    overlay.style.display = 'flex'; // Show the overlay
    content.style.display = 'none'; // Hide the content
  }
}

// Run the check on load and resize
window.addEventListener('load', checkAspectRatio);
window.addEventListener('resize', checkAspectRatio);

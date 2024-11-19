function checkAspectRatio() {
  const overlay = document.getElementById('overlay');

  // Check if the aspect ratio matches a mobile device (height > width)
  if (window.innerHeight > window.innerWidth) {
    overlay.style.display = 'none'; // Hide the overlay
  } else {
    overlay.style.display = 'flex'; // Show the overlay
  }
}

// Run the check on load and resize
window.addEventListener('load', checkAspectRatio);
window.addEventListener('resize', checkAspectRatio);

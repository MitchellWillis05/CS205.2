const hamburgerMenu = document.getElementById('hamburger-menu');
const sideMenu = document.getElementById('side-menu');
const closeMenu = document.getElementById('close-menu');

// Open the side menu
hamburgerMenu.addEventListener('click', () => {
  sideMenu.style.right = '0'; // Slide in
});

// Close the side menu
closeMenu.addEventListener('click', () => {
  sideMenu.style.right = '-100%'; // Slide out
});

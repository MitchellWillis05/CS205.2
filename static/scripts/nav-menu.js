const hamburgerMenu = document.getElementById('hamburger-menu');
const sideMenu = document.getElementById('side-menu');
const closeMenu = document.getElementById('close-menu');

hamburgerMenu.addEventListener('click', () => {
  sideMenu.style.right = '0';
});

closeMenu.addEventListener('click', () => {
  sideMenu.style.right = '-100%';
});

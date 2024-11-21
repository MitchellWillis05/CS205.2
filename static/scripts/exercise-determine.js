AOS.init({
    once: true
});

function initDropdown(toggleId, menuId) {
    const dropdownToggle = document.getElementById(toggleId);
    const dropdownMenu = document.getElementById(menuId);

    dropdownToggle.addEventListener('click', () => {
        dropdownMenu.classList.toggle('show'); // Toggle the visibility of the dropdown menu
    });

    document.addEventListener('click', (event) => {
        if (!dropdownToggle.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.classList.remove('show');
        }
    });

    dropdownMenu.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', (event) => {
            dropdownToggle.innerHTML = `${event.target.innerText} <i class='bx bx-dots-vertical-rounded'></i>`;
            dropdownMenu.classList.remove('show');
        });
    });
}

initDropdown('dropdown-toggle-1', 'dropdown-menu-1');
initDropdown('dropdown-toggle-2', 'dropdown-menu-2');

const nav = document.getElementById('nav');
console.log(nav);
console.log("hello");
if (nav) {
    nav.style.display = 'none';
}

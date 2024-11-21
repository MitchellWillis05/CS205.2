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

async function exerciseConfirm() {
    const equipment = document.getElementById("dropdown-toggle-1")
    const bodypart = document.getElementById("dropdown-toggle-2")
    const aEquipment = equipment.innerText;
    const aBodypart = bodypart.innerText;

    try
    {
        const response = await fetch('/exercises',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ equipment: aEquipment, bodypart: aBodypart}),
            });

            if (response.ok)
            {
                const result = await response.json();
                window.location.href = "/exercise-result";
            }
            else
            {
                const result = await response.json();
                triggerFlash();
            }
    }

    catch (error)
    {
        console.error('Error submitting entry:', error);
        error.innerText = 'An error occurred, please try again.';
    }
}

const nav = document.getElementById('nav');
console.log(nav);
console.log("hello");
if (nav) {
    nav.style.display = 'none';
}

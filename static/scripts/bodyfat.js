AOS.init({
    once:true
});

const bodyfat = document.getElementById('fat-input');


function incrementFat() {
    bodyfat.stepUp();
}

function decrementFat() {
    bodyfat.stepDown();
}

async function submitBodyFat() {
    const aBodyfat = bodyfat.value;

    try
    {
        const response = await fetch('/bodyfat',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ bodyfat: aBodyfat}),
            });

            if (response.ok)
            {
                const result = await response.json();
                window.location.href = "/result";
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

function triggerFlash() {
  const textElement = document.getElementById('subheading');

  textElement.classList.add('flash-once');

  textElement.addEventListener('animationend', function handleAnimationEnd() {
    textElement.classList.remove('flash-once');
    textElement.removeEventListener('animationend', handleAnimationEnd);
  });
}

const nav = document.getElementById('nav');
console.log(nav);
console.log("hello");
if (nav) {
    nav.style.display = 'none';
}
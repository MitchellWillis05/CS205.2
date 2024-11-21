AOS.init({
    once:true
});

const height = document.getElementById('height-input');
const weight = document.getElementById('weight-input');

function incrementHeight() {
    height.stepUp();
}

function decrementHeight() {
    height.stepDown();
}

function incrementWeight() {
    weight.stepUp();
}

function decrementWeight() {
    weight.stepDown();
}

async function submitHeightWeight() {
    const aWeight = weight.value;
    const aHeight = height.value;

    try
    {
        const response = await fetch('/continue',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ weight: aWeight, height: aHeight }),
            });

            if (response.ok)
            {
                const result = await response.json();
                window.location.href = "/bodyfat";
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
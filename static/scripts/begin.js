AOS.init({
    once:true
});

const input = document.getElementById('number-input');
const maleButton = document.getElementById('male-button');
const femaleButton = document.getElementById('female-button');

function incrementNumber() {
    input.stepUp();
    validateNumber();
}

function decrementNumber() {
    input.stepDown();
    validateNumber();
}

function validateNumber() {
    if (input.value < 0) {
        input.value = 0;
    } else if (input.value > 100) {
        input.value = 100;
    }
}

function selectGender(buttonToSelect, buttonToDeselect) {
    // Only toggle if the button isn't already selected
    if (!buttonToSelect.classList.contains('active')) {
        buttonToSelect.classList.add('active');
        buttonToDeselect.classList.remove('active');
    }
}

maleButton.addEventListener('click', () => {
    selectGender(maleButton, femaleButton);
});

femaleButton.addEventListener('click', () => {
    selectGender(femaleButton, maleButton);
});

async function submitAgeGender() {
    const age = input.value;
    const gender = maleButton.classList.contains('active') ? 'male' : (femaleButton.classList.contains('active') ? 'female' : null);


    try
    {
        const response = await fetch('/begin',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ age: age, gender: gender }),
            });

            if (response.ok)
            {
                const result = await response.json();
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
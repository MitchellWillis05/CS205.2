AOS.init({
    once:true
});

function backx(){
    window.location.href = "/exercises";
}

function homex(){
    window.location.href = "/";
}

const nav = document.getElementById('nav');
console.log(nav);
console.log("hello");
if (nav) {
    nav.style.display = 'none';
}
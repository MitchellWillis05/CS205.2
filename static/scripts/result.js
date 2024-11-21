AOS.init({
    once:true
});

function resultconfirm(){
    window.location.href = "/exercises";
}

const nav = document.getElementById('nav');
console.log(nav);
console.log("hello");
if (nav) {
    nav.style.display = 'none';
}
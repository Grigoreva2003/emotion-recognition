document.querySelectorAll(".nav-link").forEach((link) => {
    if (link.href === window.location.href) {
        link.classList.add("active");
        link.setAttribute("aria-current", "page");
    } else {
        link.classList.remove("active");
    }
});

var span = document.getElementById("label");
function changeFilename(myFile) {
    var file = myFile.files[0];
    span.textContent = file.name;
}

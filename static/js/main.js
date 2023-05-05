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


var name = "diagram";
var field0 = "0 ";
var previousPercentage = 0;
var filed3 = " 100";

for (var i = 1; i < 8; i++) {
    var obj = document.getElementById(name + i);
    if (obj) {
        var curPercentage = obj.dataset.percentage;
        obj.style.setProperty("stroke-dasharray",
            field0 + previousPercentage + " " + curPercentage + filed3);
        previousPercentage = Number(curPercentage) + Number(previousPercentage);
    }
}
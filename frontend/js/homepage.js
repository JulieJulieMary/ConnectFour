
window.onload = function () {
    const titleText = document.querySelector('.title_text');
    const subtitleText = document.querySelector('.subtitle_text');

    let colorIndex = 0;
    const colors = ['pink', 'orange', 'white'];

    function changeColor() {
        this.style.color = colors[colorIndex];
        colorIndex = (colorIndex + 1) % colors.length;
    }

    titleText.addEventListener('click', changeColor);
    subtitleText.addEventListener('click', changeColor);
}

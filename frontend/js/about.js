window.onload = function () {
    draw();

    let x = Number(localStorage.getItem("moves"));

    if (x) {
        localStorage.setItem("moves", "0");
    }

    function draw() {
        const canvas = document.getElementById("canvas");
        if (canvas.getContext) {
            const ctx = canvas.getContext("2d");
            const rowHeight = canvas.height / 7;
            const colWidth = canvas.width / 6;

            // Fill the canvas with light blue
            ctx.fillStyle = "lightblue";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Center the board on the page
            const boardWidth = colWidth * 6;
            const boardHeight = rowHeight * 7;
            const offsetX = (canvas.width - boardWidth) / 2;
            const offsetY = (canvas.height - boardHeight) / 2;

            // Draw rows
            // Draw rows
            ctx.strokeStyle = "pink";
            for (let i = 0; i < 7; i++) {
                ctx.beginPath();
                ctx.moveTo(offsetX, offsetY + i * rowHeight);
                ctx.lineTo(offsetX + boardWidth, offsetY + i * rowHeight);
                ctx.stroke();
            }

            // Draw columns
            for (let i = 0; i < 6; i++) {
                ctx.beginPath();
                ctx.moveTo(offsetX + i * colWidth, offsetY);
                ctx.lineTo(offsetX + i * colWidth, offsetY + boardHeight);
                ctx.stroke();
            }

            // Draw white circles in each cell
            ctx.fillStyle = "white";
            for (let i = 0; i < 6; i++) {
                for (let j = 0; j < 7; j++) {
                    ctx.beginPath();
                    ctx.arc(
                        offsetX + i * colWidth + colWidth / 2,
                        offsetY + j * rowHeight + rowHeight / 2,
                        Math.min(colWidth, rowHeight) / 2 - 2,
                        0,
                        2 * Math.PI
                    );
                    ctx.fill();
                }
            }
        }
    }

    setInterval(function () {

        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        const rowHeight = canvas.height / 7;
        const colWidth = canvas.width / 6;
        const offsetX = (canvas.width - colWidth * 6) / 2;
        const offsetY = (canvas.height - rowHeight * 7) / 2;

        // Get a random cell
        const randomCol = Math.floor(Math.random() * 6);
        const randomRow = Math.floor(Math.random() * 7);

        // Draw circle in the random cell
        ctx.fillStyle = "pink";
        ctx.beginPath();
        ctx.arc(
            offsetX + randomCol * colWidth + colWidth / 2,
            offsetY + randomRow * rowHeight + rowHeight / 2,
            Math.min(colWidth, rowHeight) / 2 - 2,
            0,
            2 * Math.PI
        );
        ctx.fill();

        // Remove the  circle after 2 seconds
        setTimeout(function () {
            ctx.fillStyle = "white";
            ctx.beginPath();
            ctx.arc(
                offsetX + randomCol * colWidth + colWidth / 2,
                offsetY + randomRow * rowHeight + rowHeight / 2,
                Math.min(colWidth, rowHeight) / 2 - 2,
                0,
                2 * Math.PI
            );
            ctx.fill();
        }, 2000);

        
        let x = Number(localStorage.getItem("moves"));
        localStorage.setItem("moves", x + 1);
       

        const moves = Number(localStorage.getItem("moves"));
        const display = document.getElementById("moves-display");
        display.textContent = `Moves: ${moves}`;

    }, 2000);

}